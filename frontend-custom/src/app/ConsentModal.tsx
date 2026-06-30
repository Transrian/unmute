"use client";
import { useEffect, useState } from "react";
import SquareButton from "./SquareButton";

export const RECORDING_CONSENT_STORAGE_KEY = "recordingConsent";

export function useConsentState(storageKey: string) {
  const [consentGiven, setConsentGiven] = useState<boolean | null>(false);
  const [consentLoaded, setConsentLoaded] = useState<boolean>(false);

  useEffect(() => {
    const consent = localStorage.getItem(storageKey);
    setConsentGiven(consent == null ? null : consent === "true");
    setConsentLoaded(true);

    // Listen for localStorage changes (in case consent is given on another tab/page)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === storageKey) {
        setConsentGiven(e.newValue === "true");
      }
    };

    window.addEventListener("storage", handleStorageChange);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
    };
  }, [storageKey]);

  const setConsent = (to: boolean | null) => {
    if (to != null) {
      localStorage.setItem(storageKey, "" + to);
    } else {
      localStorage.removeItem(storageKey);
    }
    setConsentGiven(to);
  };

  return {
    consentGiven,
    consentLoaded, // useful to avoid hydration mismatches
    setConsent,
  };
}

export default function ConsentModal() {
  const {
    consentGiven: recordingConsentGiven,
    consentLoaded: recordingConsentLoaded,
    setConsent: setRecordingConsent,
  } = useConsentState(RECORDING_CONSENT_STORAGE_KEY);
  const [recordingChecked, setRecordingChecked] = useState(true);

  useEffect(() => {
    // Only update checkbox if consent is not null (user has made a choice)
    if (recordingConsentLoaded && recordingConsentGiven !== null) {
      setRecordingChecked(recordingConsentGiven === true);
    }
  }, [recordingConsentGiven, recordingConsentLoaded]);

  if (!recordingConsentLoaded) {
    return null; // Wait until consent state is loaded
  }

  if (recordingConsentGiven !== null) {
    // User has already made a choice
    return null;
  }

  // consent is null, meaning it hasn't been given or declined yet
  return (
    <div className="fixed bottom-0 left-0 right-0 bg-gray border-t border-green shadow-lg z-50">
      <div className="max-w-7xl mx-auto p-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex-1 text-sm text-textgray">
            <div className="flex items-center mt-2">
              <input
                id="recording-consent-checkbox"
                type="checkbox"
                checked={recordingChecked}
                onChange={(e) => setRecordingChecked(e.target.checked)}
                className="mr-2"
              />
              <label htmlFor="recording-consent-checkbox">
                Allow us to record the transcript of the conversation (your
                voice will not be stored) to help our non-profit research
              </label>
            </div>
          </div>

          <div className="flex flex-row gap-2 w-full sm:w-auto justify-center">
            <SquareButton
              kind="primary"
              onClick={() => {
                setRecordingConsent(recordingChecked);
              }}
            >
              Accept
            </SquareButton>
            <SquareButton
              kind="secondary"
              onClick={() => {
                setRecordingConsent(false);
              }}
            >
              Decline
            </SquareButton>
          </div>
        </div>
      </div>
    </div>
  );
}

import { useEffect, useState } from "react";

const useKeyboardShortcuts = () => {
  const [showSubtitles, setShowSubtitles] = useState(false);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const activeElement = document.activeElement;
      // Don't toggle if the active element is an input field
      const isInputField =
        activeElement &&
        (activeElement.tagName === "INPUT" ||
          activeElement.tagName === "TEXTAREA" ||
          activeElement.getAttribute("contenteditable") === "true");

      if (!isInputField && (event.key === "S" || event.key === "s")) {
        setShowSubtitles((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [setShowSubtitles]);

  return { showSubtitles };
};

export default useKeyboardShortcuts;

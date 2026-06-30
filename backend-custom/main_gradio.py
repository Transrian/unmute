from fastrtc import Stream, get_hf_turn_credentials

from unmute.unmute_handler import UnmuteHandler

if __name__ == "__main__":
    rtc_configuration = get_hf_turn_credentials()

    stream = Stream(
        handler=UnmuteHandler(),
        modality="audio",
        mode="send-receive",
        rtc_configuration=rtc_configuration,
        concurrency_limit=1,
    )

    demo = stream.ui
    demo.launch(debug=False)

const uploadForm = document.getElementById("upload-form");
const audioInput = document.getElementById("audio-input");

const uploadZone = document.getElementById("upload-zone");

const selectedFile = document.getElementById("selected-file");
const fileName = document.getElementById("file-name");
const removeFile = document.getElementById("remove-file");

const analyzeButton = document.getElementById("analyze-button");

const processing = document.getElementById("processing");

const errorPanel = document.getElementById("error-panel");
const errorMessage = document.getElementById("error-message");

const results = document.getElementById("results");
const speakerList = document.getElementById("speaker-list");
const transcript = document.getElementById("transcript");

const resetButton = document.getElementById("reset-button");

const audioPlayer = document.getElementById("audio-player");
const audioElement = document.getElementById("audio");
const audioFileName = document.getElementById("audio-file-name");


let selectedAudio = null;
let analysisResult = null;
let activeSpeaker = null;


/* ==========================================================================
   File selection
   ========================================================================== */

audioInput.addEventListener(
    "change",
    () => {
        const file = audioInput.files[0];

        if (!file) {
            clearSelectedFile();
            return;
        }

        selectedAudio = file;

        fileName.textContent = file.name;

        selectedFile.hidden = false;

        analyzeButton.disabled = false;

        hideError();

        results.hidden = true;
    }
);


/* ==========================================================================
   Upload zone
   ========================================================================== */

uploadZone.addEventListener(
    "dragover",
    (event) => {
        event.preventDefault();

        uploadZone.classList.add(
            "dragging"
        );
    }
);


uploadZone.addEventListener(
    "dragleave",
    () => {
        uploadZone.classList.remove(
            "dragging"
        );
    }
);


uploadZone.addEventListener(
    "drop",
    (event) => {
        event.preventDefault();

        uploadZone.classList.remove(
            "dragging"
        );

        const file =
            event.dataTransfer.files[0];

        if (!file) {
            return;
        }

        selectedAudio = file;

        fileName.textContent =
            file.name;

        selectedFile.hidden = false;

        analyzeButton.disabled = false;

        hideError();

        results.hidden = true;
    }
);


/* ==========================================================================
   Remove file
   ========================================================================== */

removeFile.addEventListener(
    "click",
    (event) => {
        event.preventDefault();
        event.stopPropagation();

        clearSelectedFile();
    }
);


function clearSelectedFile() {

    selectedAudio = null;

    audioInput.value = "";

    selectedFile.hidden = true;

    fileName.textContent = "";

    analyzeButton.disabled = true;
}


/* ==========================================================================
   Upload + analysis
   ========================================================================== */

uploadForm.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();

        if (!selectedAudio) {

            showError(
                "Please select an audio file first."
            );

            return;
        }


        setProcessingState(
            true
        );


        const formData =
            new FormData();


        formData.append(
            "audio",
            selectedAudio
        );


        try {

            const response =
                await fetch(
                    "/api/analyze",
                    {
                        method: "POST",
                        body: formData,
                    }
                );


            let data = null;


            try {

                data =
                    await response.json();

            } catch {

                data = null;
            }


            /* --------------------------------------------------------------
               Success
               -------------------------------------------------------------- */

            if (
                response.ok &&
                data &&
                data.success === true
            ) {

                analysisResult =
                    data.result;

                renderResults(
                    analysisResult
                );

                return;
            }


            /* --------------------------------------------------------------
               Backend error
               -------------------------------------------------------------- */

            showError(
                getErrorMessage(
                    response,
                    data
                )
            );

        } catch (error) {

            console.error(
                "Analysis request failed:",
                error
            );

            showError(
                "Unable to connect to the analysis service."
            );

        } finally {

            setProcessingState(
                false
            );
        }
    }
);


/* ==========================================================================
   Backend error messages
   ========================================================================== */

function getErrorMessage(
    response,
    data
) {

    if (
        data &&
        typeof data.error === "string" &&
        data.error.trim()
    ) {

        return data.error;
    }


    switch (
        response.status
    ) {

        case 400:
            return (
                "The uploaded audio could not be processed."
            );

        case 413:
            return (
                "The uploaded file is too large."
            );

        case 503:
            return (
                "Transcription is temporarily unavailable. "
                + "Please try the analysis again."
            );

        case 500:
            return (
                "The audio could not be analyzed. "
                + "Please try again."
            );

        default:
            return (
                "Something went wrong while analyzing "
                + "the audio. Please try again."
            );
    }
}


/* ==========================================================================
   Processing state
   ========================================================================== */

function setProcessingState(
    isProcessing
) {

    processing.hidden =
        !isProcessing;


    if (isProcessing) {

        results.hidden = true;

        analyzeButton.disabled = true;

        hideError();

    } else {

        analyzeButton.disabled =
            !selectedAudio;
    }
}


/* ==========================================================================
   Error
   ========================================================================== */

function showError(
    message
) {

    errorMessage.textContent =
        message;

    errorPanel.hidden = false;

    results.hidden = true;
}


function hideError() {

    errorPanel.hidden = true;

    errorMessage.textContent = "";
}


/* ==========================================================================
   Results
   ========================================================================== */

function renderResults(
    result
) {

    hideError();


    /*
     * Keep the uploaded source audio available
     * to the result view.
     */

    if (selectedAudio) {

        const objectURL =
            URL.createObjectURL(
                selectedAudio
            );

        audioElement.src =
            objectURL;

        audioFileName.textContent =
            selectedAudio.name;

        audioPlayer.hidden = false;
    }


    renderSpeakers(
        getSpeakers(result)
    );


    renderTranscript(
        getTranscript(result)
    );


    results.hidden = false;


    results.scrollIntoView(
        {
            behavior: "smooth",
            block: "start",
        }
    );
}


/* ==========================================================================
   Result data helpers
   ========================================================================== */

function getSpeakers(
    result
) {

    if (
        Array.isArray(
            result?.speakers
        )
    ) {

        return result.speakers;
    }


    const segments =
        getTranscript(result);


    const speakers =
        segments
            .map(
                segment =>
                    segment.speaker
            )
            .filter(
                Boolean
            );


    return [
        ...new Set(
            speakers
        ),
    ];
}


function getTranscript(
    result
) {

    if (
        Array.isArray(
            result?.transcript
        )
    ) {

        return result.transcript;
    }


    /*
     * Some pipeline versions may return
     * the transcript under another structure.
     */

    if (
        Array.isArray(
            result?.segments
        )
    ) {

        return result.segments;
    }


    return [];
}


/* ==========================================================================
   Speakers
   ========================================================================== */

function renderSpeakers(
    speakers
) {

    speakerList.innerHTML = "";

    activeSpeaker = null;


    if (!speakers.length) {
        return;
    }


    /* --------------------------------------------------------------
       All speakers
       -------------------------------------------------------------- */

    const allButton =
        document.createElement(
            "button"
        );

    allButton.type =
        "button";

    allButton.className =
        "speaker speaker-filter active";

    allButton.textContent =
        "ALL";

    allButton.dataset.speaker =
        "all";


    allButton.addEventListener(
        "click",
        () => {

            setActiveSpeaker(
                null
            );
        }
    );


    speakerList.appendChild(
        allButton
    );


    /* --------------------------------------------------------------
       Individual speakers
       -------------------------------------------------------------- */

    for (
        const speaker of speakers
    ) {

        const button =
            document.createElement(
                "button"
            );


        button.type =
            "button";

        button.className =
            "speaker speaker-filter";

        button.textContent =
            speaker;

        button.dataset.speaker =
            speaker;


        button.addEventListener(
            "click",
            () => {

                setActiveSpeaker(
                    speaker
                );
            }
        );


        speakerList.appendChild(
            button
        );
    }
}


/* ==========================================================================
   Speaker filtering
   ========================================================================== */

function setActiveSpeaker(
    speaker
) {

    activeSpeaker =
        speaker;


    const buttons =
        speakerList.querySelectorAll(
            ".speaker-filter"
        );


    buttons.forEach(
        button => {

            const buttonSpeaker =
                button.dataset.speaker;


            const isActive =
                (
                    speaker === null &&
                    buttonSpeaker === "all"
                ) ||
                buttonSpeaker === speaker;


            button.classList.toggle(
                "active",
                isActive
            );
        }
    );


    renderTranscript(
        getTranscript(
            analysisResult
        )
    );
}


/* ==========================================================================
   Transcript
   ========================================================================== */

function renderTranscript(
    segments
) {

    transcript.innerHTML = "";


    const visibleSegments =
        activeSpeaker === null
            ? segments
            : segments.filter(
                segment =>
                    segment.speaker ===
                    activeSpeaker
            );


    if (!visibleSegments.length) {

        const empty =
            document.createElement(
                "p"
            );


        empty.className =
            "transcript-empty";


        empty.textContent =
            activeSpeaker
                ? `No transcript segments found for ${activeSpeaker}.`
                : "No transcript segments were returned.";


        transcript.appendChild(
            empty
        );

        return;
    }


    for (
        const segment of visibleSegments
    ) {

        const row =
            document.createElement(
                "article"
            );


        row.className =
            "transcript-segment";


        /*
         * Timestamp
         */

        const time =
            document.createElement(
                "button"
            );


        time.type =
            "button";

        time.className =
            "transcript-time";

        time.textContent =
            `${formatTime(segment.start)} → ` +
            `${formatTime(segment.end)}`;

        time.title =
            "Jump to this timestamp";


        time.addEventListener(
            "click",
            () => {

                seekAudio(
                    segment.start
                );
            }
        );


        /*
         * Speaker
         */

        const speaker =
            document.createElement(
                "div"
            );


        speaker.className =
            "transcript-speaker";


        speaker.textContent =
            segment.speaker ||
            "unknown";


        /*
         * Text
         */

        const text =
            document.createElement(
                "div"
            );


        text.className =
            "transcript-text";


        text.textContent =
            segment.text ||
            "";


        /*
         * Row click also seeks.
         */

        row.addEventListener(
            "click",
            (event) => {

                if (
                    event.target.closest(
                        ".transcript-time"
                    )
                ) {
                    return;
                }


                seekAudio(
                    segment.start
                );
            }
        );


        row.tabIndex =
            0;


        row.addEventListener(
            "keydown",
            (event) => {

                if (
                    event.key === "Enter" ||
                    event.key === " "
                ) {

                    event.preventDefault();

                    seekAudio(
                        segment.start
                    );
                }
            }
        );


        row.appendChild(
            time
        );

        row.appendChild(
            speaker
        );

        row.appendChild(
            text
        );


        transcript.appendChild(
            row
        );
    }
}


/* ==========================================================================
   Audio seeking
   ========================================================================== */

function seekAudio(
    seconds
) {

    if (
        !audioElement ||
        !Number.isFinite(
            Number(seconds)
        )
    ) {
        return;
    }


    audioElement.currentTime =
        Number(seconds);


    audioElement.play().catch(
        () => {
            /*
             * Browsers may block automatic
             * playback. The seek still works.
             */
        }
    );
}


/* ==========================================================================
   Time formatting
   ========================================================================== */

function formatTime(
    seconds
) {

    const value =
        Math.max(
            0,
            Number(seconds) || 0
        );


    const minutes =
        Math.floor(
            value / 60
        );


    const remaining =
        value % 60;


    return (
        `${String(minutes).padStart(2, "0")}:` +
        `${remaining
            .toFixed(1)
            .padStart(4, "0")}`
    );
}


/* ==========================================================================
   Reset
   ========================================================================== */

resetButton.addEventListener(
    "click",
    () => {

        if (
            audioElement.src
        ) {

            URL.revokeObjectURL(
                audioElement.src
            );
        }


        audioElement.pause();

        audioElement.removeAttribute(
            "src"
        );

        audioElement.load();


        analysisResult = null;

        activeSpeaker = null;


        results.hidden = true;

        hideError();

        clearSelectedFile();


        window.scrollTo(
            {
                top: 0,
                behavior: "smooth",
            }
        );
    }
);

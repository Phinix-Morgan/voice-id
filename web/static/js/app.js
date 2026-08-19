const uploadForm = document.getElementById("upload-form");
const audioInput = document.getElementById("audio-input");
const uploadZone = document.getElementById("upload-zone");

const selectedFile = document.getElementById("selected-file");
const fileName = document.getElementById("file-name");
const removeFile = document.getElementById("remove-file");

const analyzeButton = document.getElementById("analyze-button");

const processing = document.getElementById("processing");
const processingTitle = document.getElementById("processing-title");
const processingSubtitle = document.getElementById("processing-subtitle");
const processingSteps = [...document.querySelectorAll(".processing-step")];

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
let activeSegment = null;
let audioObjectURL = null;
let processingTimer = null;
let processingStepIndex = 0;


/* ==========================================================================
   File selection
   ========================================================================== */

audioInput.addEventListener("change", () => {
    const file = audioInput.files[0];

    if (!file) {
        clearSelectedFile();
        return;
    }

    selectAudioFile(file);
});


function selectAudioFile(file) {
    selectedAudio = file;

    fileName.textContent = file.name;
    selectedFile.hidden = false;

    analyzeButton.disabled = false;
    hideError();

    results.hidden = true;
}


/* ==========================================================================
   Drag and drop
   ========================================================================== */

["dragenter", "dragover"].forEach((eventName) => {
    uploadZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        uploadZone.classList.add("dragging");
    });
});


["dragleave", "drop"].forEach((eventName) => {
    uploadZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        uploadZone.classList.remove("dragging");
    });
});


uploadZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];

    if (!file) {
        return;
    }

    selectAudioFile(file);

    /*
     * Keep the native input in sync where possible.
     * DataTransfer is supported by modern browsers.
     */
    try {
        const transfer = new DataTransfer();
        transfer.items.add(file);
        audioInput.files = transfer.files;
    } catch {
        /* The selectedAudio state is sufficient for submission. */
    }
});


/* ==========================================================================
   Remove file
   ========================================================================== */

removeFile.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();

    clearSelectedFile();
});


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

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!selectedAudio) {
        showError("Please select an audio file first.");
        return;
    }

    setProcessingState(true);

    const formData = new FormData();
    formData.append("audio", selectedAudio);

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData,
        });

        let data = null;

        try {
            data = await response.json();
        } catch {
            data = null;
        }

        if (response.ok && data && data.success === true) {
            analysisResult = data.result;
            renderResults(analysisResult);
            return;
        }

        showError(getErrorMessage(response, data));
    } catch (error) {
        console.error("Analysis request failed:", error);

        showError(
            "Unable to connect to the analysis service."
        );
    } finally {
        setProcessingState(false);
    }
});


/* ==========================================================================
   Backend error messages
   ========================================================================== */

function getErrorMessage(response, data) {
    if (
        data &&
        typeof data.error === "string" &&
        data.error.trim()
    ) {
        return data.error;
    }

    switch (response.status) {
        case 400:
            return "The uploaded audio could not be processed.";

        case 413:
            return "The uploaded file is too large.";

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

function setProcessingState(isProcessing) {
    processing.hidden = !isProcessing;

    if (isProcessing) {
        results.hidden = true;
        analyzeButton.disabled = true;
        analyzeButton.classList.add("is-loading");
        hideError();
        startProcessingAnimation();
    } else {
        analyzeButton.classList.remove("is-loading");
        analyzeButton.disabled = !selectedAudio;
        stopProcessingAnimation();
    }
}


function startProcessingAnimation() {
    processingStepIndex = 0;

    updateProcessingStep();

    clearInterval(processingTimer);

    processingTimer = setInterval(() => {
        processingStepIndex =
            (processingStepIndex + 1) % processingSteps.length;

        updateProcessingStep();
    }, 1800);
}


function updateProcessingStep() {
    const stages = [
        {
            title: "Identifying speakers",
            subtitle: "Matching voices against the known speaker profiles.",
        },
        {
            title: "Transcribing audio",
            subtitle: "Converting speech into timestamped text.",
        },
        {
            title: "Aligning conversation",
            subtitle: "Linking transcript segments to their speakers.",
        },
    ];

    const stage = stages[processingStepIndex];

    if (!stage) {
        return;
    }

    processingTitle.textContent = stage.title;
    processingSubtitle.textContent = stage.subtitle;

    processingSteps.forEach((step, index) => {
        step.classList.toggle(
            "active",
            index === processingStepIndex
        );
    });
}


function stopProcessingAnimation() {
    clearInterval(processingTimer);
    processingTimer = null;
}


/* ==========================================================================
   Error
   ========================================================================== */

function showError(message) {
    errorMessage.textContent = message;
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

function renderResults(result) {
    hideError();

    if (selectedAudio) {
        revokeAudioObjectURL();

        audioObjectURL = URL.createObjectURL(selectedAudio);

        audioElement.src = audioObjectURL;
        audioFileName.textContent = selectedAudio.name;
        audioPlayer.hidden = false;
    }

    renderSpeakers(getSpeakers(result));
    renderTranscript(getTranscript(result));

    results.hidden = false;

    requestAnimationFrame(() => {
        results.scrollIntoView({
            behavior: "smooth",
            block: "start",
        });
    });
}


/* ==========================================================================
   Result data helpers
   ========================================================================== */

function getSpeakers(result) {
    if (Array.isArray(result?.speakers)) {
        return result.speakers;
    }

    const segments = getTranscript(result);

    return [
        ...new Set(
            segments
                .map((segment) => segment.speaker)
                .filter(Boolean)
        ),
    ];
}


function getTranscript(result) {
    if (Array.isArray(result?.transcript)) {
        return result.transcript;
    }

    if (Array.isArray(result?.segments)) {
        return result.segments;
    }

    return [];
}


/* ==========================================================================
   Speakers
   ========================================================================== */

function renderSpeakers(speakers) {
    speakerList.innerHTML = "";
    activeSpeaker = null;

    if (!speakers.length) {
        return;
    }

    const allButton = document.createElement("button");

    allButton.type = "button";
    allButton.className = "speaker speaker-filter active";
    allButton.textContent = "ALL";
    allButton.dataset.speaker = "all";

    allButton.addEventListener("click", () => {
        setActiveSpeaker(null);
    });

    speakerList.appendChild(allButton);

    for (const speaker of speakers) {
        const button = document.createElement("button");

        button.type = "button";
        button.className = "speaker speaker-filter";
        button.textContent = speaker;
        button.dataset.speaker = speaker;

        button.addEventListener("click", () => {
            setActiveSpeaker(speaker);
        });

        speakerList.appendChild(button);
    }
}


function setActiveSpeaker(speaker) {
    activeSpeaker = speaker;
    activeSegment = null;

    const buttons =
        speakerList.querySelectorAll(".speaker-filter");

    buttons.forEach((button) => {
        const buttonSpeaker = button.dataset.speaker;

        const isActive =
            (speaker === null && buttonSpeaker === "all") ||
            buttonSpeaker === speaker;

        button.classList.toggle("active", isActive);
    });

    renderTranscript(getTranscript(analysisResult));
}


/* ==========================================================================
   Transcript
   ========================================================================== */

function renderTranscript(segments) {
    transcript.innerHTML = "";
    activeSegment = null;

    const visibleSegments =
        activeSpeaker === null
            ? segments
            : segments.filter(
                (segment) =>
                    segment.speaker === activeSpeaker
            );

    if (!visibleSegments.length) {
        const empty = document.createElement("p");

        empty.className = "transcript-empty";

        empty.textContent = activeSpeaker
            ? `No transcript segments found for ${activeSpeaker}.`
            : "No transcript segments were returned.";

        transcript.appendChild(empty);

        return;
    }

    visibleSegments.forEach((segment, index) => {
        const row = document.createElement("article");

        row.className = "transcript-segment";
        row.dataset.speaker =
            segment.speaker || "unknown";

        row.dataset.start = Number(segment.start) || 0;
        row.dataset.end = Number(segment.end) || 0;

        row.style.animationDelay =
            `${Math.min(index * 35, 420)}ms`;

        /*
         * Timestamp
         */
        const time = document.createElement("button");

        time.type = "button";
        time.className = "transcript-time";
        time.textContent =
            `${formatTime(segment.start)} → ${formatTime(segment.end)}`;
        time.title = "Jump to this timestamp";

        time.addEventListener("click", (event) => {
            event.stopPropagation();
            seekAudio(segment.start);
            setActiveSegment(row);
        });

        /*
         * Speaker
         */
        const speaker = document.createElement("div");

        speaker.className = "transcript-speaker";
        speaker.textContent =
            segment.speaker || "unknown";

        /*
         * Text
         */
        const text = document.createElement("div");

        text.className = "transcript-text";
        text.textContent = segment.text || "";

        /*
         * Row click
         */
        row.addEventListener("click", () => {
            seekAudio(segment.start);
            setActiveSegment(row);
        });

        row.tabIndex = 0;

        row.addEventListener("keydown", (event) => {
            if (
                event.key === "Enter" ||
                event.key === " "
            ) {
                event.preventDefault();

                seekAudio(segment.start);
                setActiveSegment(row);
            }
        });

        row.appendChild(time);
        row.appendChild(speaker);
        row.appendChild(text);

        transcript.appendChild(row);
    });
}


function setActiveSegment(row) {
    transcript
        .querySelectorAll(".transcript-segment.active")
        .forEach((segment) => {
            segment.classList.remove("active");
        });

    row.classList.add("active");
    activeSegment = row;
}


/* ==========================================================================
   Audio seeking + active transcript tracking
   ========================================================================== */

function seekAudio(seconds) {
    const value = Number(seconds);

    if (
        !audioElement ||
        !Number.isFinite(value)
    ) {
        return;
    }

    audioElement.currentTime = value;

    audioElement.play().catch(() => {
        /* Browser autoplay restrictions are harmless here. */
    });
}


audioElement.addEventListener("timeupdate", () => {
    const currentTime = audioElement.currentTime;

    const segments =
        transcript.querySelectorAll(".transcript-segment");

    let matched = null;

    segments.forEach((row) => {
        const start = Number(row.dataset.start);
        const end = Number(row.dataset.end);

        const isActive =
            currentTime >= start &&
            currentTime < end;

        row.classList.toggle("active", isActive);

        if (isActive) {
            matched = row;
        }
    });

    activeSegment = matched;

    /*
     * Keep the active line visible without forcing scroll on every
     * timeupdate. Only scroll when playback enters a new segment.
     */
    if (
        matched &&
        matched !== lastAutoScrolledSegment
    ) {
        lastAutoScrolledSegment = matched;

        matched.scrollIntoView({
            behavior: "smooth",
            block: "nearest",
        });
    }
});


let lastAutoScrolledSegment = null;


/* ==========================================================================
   Time formatting
   ========================================================================== */

function formatTime(seconds) {
    const value = Math.max(
        0,
        Number(seconds) || 0
    );

    const minutes = Math.floor(value / 60);
    const remaining = value % 60;

    return (
        `${String(minutes).padStart(2, "0")}:` +
        `${remaining.toFixed(1).padStart(4, "0")}`
    );
}


/* ==========================================================================
   Reset
   ========================================================================== */

resetButton.addEventListener("click", () => {
    audioElement.pause();

    audioElement.removeAttribute("src");
    audioElement.load();

    revokeAudioObjectURL();

    analysisResult = null;
    activeSpeaker = null;
    activeSegment = null;
    lastAutoScrolledSegment = null;

    results.hidden = true;

    hideError();
    clearSelectedFile();

    window.scrollTo({
        top: 0,
        behavior: "smooth",
    });
});


function revokeAudioObjectURL() {
    if (!audioObjectURL) {
        return;
    }

    URL.revokeObjectURL(audioObjectURL);
    audioObjectURL = null;
}


/* ==========================================================================
   Initial state
   ========================================================================== */

processing.hidden = true;
results.hidden = true;
errorPanel.hidden = true;
audioPlayer.hidden = false;

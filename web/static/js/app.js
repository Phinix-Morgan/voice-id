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
   Live microphone recorder
   ========================================================================== */

const uploadTab = document.getElementById("upload-tab");
const recordTab = document.getElementById("record-tab");
const uploadPanel = document.getElementById("upload-panel");
const recordPanel = document.getElementById("record-panel");

const recorder = document.getElementById("recorder");
const recordButton = document.getElementById("record-button");
const clearRecordingButton = document.getElementById("clear-recording");
const recorderStatusText = document.getElementById("recorder-status-text");
const recorderAction = document.getElementById("recorder-action");
const recorderHint = document.getElementById("recorder-hint");
const recordingTime = document.getElementById("recording-time");
const recordingWaveform = document.getElementById("recording-waveform");
const recordingPreview = document.getElementById("recording-preview");
const recordingPreviewAudio = document.getElementById("recording-preview-audio");
const recordingFileName = document.getElementById("recording-file-name");

let activeSource = "upload";
let mediaRecorder = null;
let mediaStream = null;
let audioContext = null;
let analyser = null;
let animationFrame = null;
let recordingStartedAt = null;
let recordingTimer = null;
let recordingChunks = [];
let recordingObjectURL = null;

uploadTab?.addEventListener("click", () => switchSource("upload"));
recordTab?.addEventListener("click", () => switchSource("record"));

function switchSource(source) {
    if (activeSource === source) return;

    if (isRecording()) stopRecording();

    activeSource = source;
    const isUpload = source === "upload";

    uploadTab?.classList.toggle("active", isUpload);
    recordTab?.classList.toggle("active", !isUpload);
    uploadTab?.setAttribute("aria-selected", String(isUpload));
    recordTab?.setAttribute("aria-selected", String(!isUpload));

    if (uploadPanel) {
        uploadPanel.hidden = !isUpload;
        uploadPanel.classList.toggle("active", isUpload);
    }

    if (recordPanel) {
        recordPanel.hidden = isUpload;
        recordPanel.classList.toggle("active", !isUpload);
    }

    hideError();
}

function isRecording() {
    return mediaRecorder?.state === "recording";
}

recordButton?.addEventListener("click", async () => {
    if (isRecording()) {
        stopRecording();
    } else {
        await startRecording();
    }
});

async function startRecording() {
    if (
        !navigator.mediaDevices?.getUserMedia ||
        typeof MediaRecorder === "undefined"
    ) {
        showError(
            "Live recording is not supported by this browser. "
            + "Please upload an audio file instead."
        );
        return;
    }

    try {
        hideError();

        if (selectedAudio) clearSelectedFile();

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

        mediaStream = stream;

        const mimeType = getSupportedRecordingMimeType();

        mediaRecorder = mimeType
            ? new MediaRecorder(stream, { mimeType })
            : new MediaRecorder(stream);

        recordingChunks = [];

        mediaRecorder.addEventListener("dataavailable", (event) => {
            if (event.data?.size > 0) recordingChunks.push(event.data);
        });

        mediaRecorder.addEventListener("stop", finishRecording, { once: true });
        mediaRecorder.start(100);

        startLiveMeter(stream);
        startRecordingClock();

        recorder?.classList.add("is-recording");
        analyzeButton.disabled = true;

        if (recordingPreview) recordingPreview.hidden = true;
        if (clearRecordingButton) clearRecordingButton.hidden = true;

        recorderStatusText.textContent = "Recording";
        recorderAction.textContent = "Stop recording";
        recorderHint.textContent =
            "Speak naturally. The waveform follows your voice.";

        recordButton?.setAttribute("aria-label", "Stop recording");
    } catch (error) {
        console.error("Microphone access failed:", error);
        stopMicrophoneStream();

        if (error?.name === "NotAllowedError") {
            showError(
                "Microphone access was denied. Allow microphone access "
                + "in your browser and try again."
            );
        } else {
            showError(
                "Unable to access the microphone. "
                + "Please check your browser permissions."
            );
        }
    }
}

function getSupportedRecordingMimeType() {
    const candidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/ogg;codecs=opus",
        "audio/ogg",
    ];

    return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function stopRecording() {
    if (!isRecording()) return;

    mediaRecorder.stop();
    stopRecordingClock();
    stopLiveMeter();

    recorder?.classList.remove("is-recording");
    recorderStatusText.textContent = "Processing recording";
    recorderAction.textContent = "Saving recording";
    recorderHint.textContent = "Preparing your audio for analysis.";
    recordButton?.setAttribute("aria-label", "Start recording");
}

function finishRecording() {
    const mimeType = mediaRecorder?.mimeType || "audio/webm";
    const blob = new Blob(recordingChunks, { type: mimeType });

    stopMicrophoneStream();
    recordingChunks = [];

    if (!blob.size) {
        resetRecorderVisuals();
        showError("No audio was captured. Please try recording again.");
        return;
    }

    const extension = getAudioExtension(mimeType);
    const filename =
        `voice-id-recording-${new Date().toISOString().replace(/[:.]/g, "-")}.${extension}`;

    selectedAudio = new File([blob], filename, {
        type: mimeType,
        lastModified: Date.now(),
    });

    fileName.textContent = filename;
    selectedFile.hidden = false;

    if (recordingObjectURL) URL.revokeObjectURL(recordingObjectURL);
    recordingObjectURL = URL.createObjectURL(blob);

    if (recordingPreviewAudio) recordingPreviewAudio.src = recordingObjectURL;
    if (recordingFileName) recordingFileName.textContent = filename;
    if (recordingPreview) recordingPreview.hidden = false;
    if (clearRecordingButton) clearRecordingButton.hidden = false;

    recorderStatusText.textContent = "Recording ready";
    recorderAction.textContent = "Ready to analyze";
    recorderHint.textContent = "Review the recording or record again.";

    analyzeButton.disabled = false;
    recorder?.classList.remove("is-recording");
}

function getAudioExtension(mimeType) {
    if (mimeType.includes("mp4")) return "m4a";
    if (mimeType.includes("ogg")) return "ogg";
    return "webm";
}

function startLiveMeter(stream) {
    stopLiveMeter();

    try {
        audioContext = new (
            window.AudioContext || window.webkitAudioContext
        )();

        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.82;

        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        drawWaveform();
    } catch (error) {
        console.warn("Live waveform unavailable:", error);
        drawIdleWaveform();
    }
}

function drawWaveform() {
    if (!analyser || !recordingWaveform) return;

    const canvas = recordingWaveform;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));

    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }

    const context = canvas.getContext("2d");
    const data = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(data);

    context.clearRect(0, 0, width, height);
    context.beginPath();
    context.lineWidth = Math.max(1, dpr * 1.15);
    context.strokeStyle = "#050505";
    context.lineCap = "round";
    context.lineJoin = "round";

    const center = height / 2;
    const amplitude = height * 0.36;

    for (let i = 0; i < data.length; i += 2) {
        const x = (i / (data.length - 1)) * width;
        const y = center + ((data[i] - 128) / 128) * amplitude;

        if (i === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
    }

    context.stroke();
    animationFrame = requestAnimationFrame(drawWaveform);
}

function drawIdleWaveform() {
    if (!recordingWaveform) return;

    const canvas = recordingWaveform;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));

    const context = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const center = height / 2;

    context.clearRect(0, 0, width, height);
    context.beginPath();
    context.lineWidth = Math.max(1, dpr);
    context.strokeStyle = "#d1d1d1";

    for (let x = 0; x <= width; x += Math.max(4, dpr * 4)) {
        const y = center
            + Math.sin((x / Math.max(1, width)) * Math.PI * 10) * dpr * 1.8;

        if (x === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
    }

    context.stroke();
}

function stopLiveMeter() {
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }

    if (audioContext) {
        audioContext.close().catch(() => {});
        audioContext = null;
    }

    analyser = null;
}

window.addEventListener("resize", () => {
    if (!isRecording()) drawIdleWaveform();
});

function startRecordingClock() {
    recordingStartedAt = performance.now();
    updateRecordingClock();
    clearInterval(recordingTimer);
    recordingTimer = setInterval(updateRecordingClock, 100);
}

function updateRecordingClock() {
    if (recordingStartedAt === null) return;

    const elapsed = (performance.now() - recordingStartedAt) / 1000;
    const minutes = Math.floor(elapsed / 60);
    const seconds = Math.floor(elapsed % 60);

    if (recordingTime) {
        recordingTime.textContent =
            `${String(minutes).padStart(2, "0")}:` +
            `${String(seconds).padStart(2, "0")}`;
    }
}

function stopRecordingClock() {
    clearInterval(recordingTimer);
    recordingTimer = null;
    recordingStartedAt = null;
}

function clearRecording() {
    if (isRecording()) stopRecording();

    stopMicrophoneStream();
    stopLiveMeter();
    stopRecordingClock();

    mediaRecorder = null;
    recordingChunks = [];

    if (recordingObjectURL) {
        URL.revokeObjectURL(recordingObjectURL);
        recordingObjectURL = null;
    }

    selectedAudio = null;
    audioInput.value = "";
    selectedFile.hidden = true;
    fileName.textContent = "";

    if (recordingPreviewAudio) {
        recordingPreviewAudio.pause();
        recordingPreviewAudio.removeAttribute("src");
        recordingPreviewAudio.load();
    }

    if (recordingPreview) recordingPreview.hidden = true;
    if (clearRecordingButton) clearRecordingButton.hidden = true;

    resetRecorderVisuals();
    analyzeButton.disabled = true;
}

clearRecordingButton?.addEventListener("click", clearRecording);

function stopMicrophoneStream() {
    if (!mediaStream) return;
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
}

function resetRecorderVisuals() {
    recorder?.classList.remove("is-recording");
    recorderStatusText.textContent = "Ready to record";
    recorderAction.textContent = "Start recording";
    recorderHint.textContent = "Allow microphone access to begin.";
    recordButton?.setAttribute("aria-label", "Start recording");

    if (recordingTime) recordingTime.textContent = "00:00";

    drawIdleWaveform();
}

drawIdleWaveform();


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

    if (activeSource === "record") {
        if (recordingPreviewAudio) {
            recordingPreviewAudio.pause();
            recordingPreviewAudio.removeAttribute("src");
            recordingPreviewAudio.load();
        }

        if (recordingPreview) recordingPreview.hidden = true;
        if (clearRecordingButton) clearRecordingButton.hidden = true;
    }

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

    if (
        selectedAudio &&
        audioElement &&
        audioFileName &&
        audioPlayer
    ) {
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


audioElement?.addEventListener("timeupdate", () => {
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
    clearRecording();

    audioElement?.pause();

    audioElement?.removeAttribute("src");
    audioElement?.load();

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

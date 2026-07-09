const fallbackSongs = [
  { name: "消愁", id: "mao_buyi_xiaochou", lineId: "demo_segment_1", text: "消愁 - demo segment", promptAudioUrl: "/static/prompts/mao_buyi_v1/mao_buyi_xiaochou_prompt.wav", features: [60, 62, 63, 67, 65, 63, 62, 60] },
  { name: "像我这样的人", id: "mao_buyi_like_me", lineId: "demo_segment_1", text: "像我这样的人 - demo segment", promptAudioUrl: "/static/prompts/mao_buyi_v1/mao_buyi_like_me_prompt.wav", features: [64, 64, 65, 67, 65, 64, 62, 60] },
  { name: "盛夏", id: "mao_buyi_summer", lineId: "demo_segment_1", text: "盛夏 - demo segment", promptAudioUrl: "/static/prompts/mao_buyi_v1/mao_buyi_summer_prompt.wav", features: [55, 60, 62, 64, 67, 64, 62, 60] },
  { name: "不染", id: "mao_buyi_unstained", lineId: "demo_segment_1", text: "不染 - demo segment", promptAudioUrl: "/static/prompts/mao_buyi_v1/mao_buyi_unstained_prompt.wav", features: [67, 69, 70, 72, 70, 69, 67, 65] },
];

const state = {
  songs: fallbackSongs,
  selected: fallbackSongs[0],
  running: false,
  recorder: null,
  recordedUrl: null,
};

const els = {
  songGrid: document.querySelector("#songGrid"),
  sendButton: document.querySelector("#sendButton"),
  randomPromptButton: document.querySelector("#randomPromptButton"),
  recordButton: document.querySelector("#recordButton"),
  clearButton: document.querySelector("#clearButton"),
  wsUrl: document.querySelector("#wsUrl"),
  catalogId: document.querySelector("#catalogId"),
  connectionStatus: document.querySelector("#connectionStatus"),
  resultTitle: document.querySelector("#resultTitle"),
  endToResult: document.querySelector("#endToResult"),
  matched: document.querySelector("#matched"),
  songId: document.querySelector("#songId"),
  confidence: document.querySelector("#confidence"),
  matchedLine: document.querySelector("#matchedLine"),
  rawJson: document.querySelector("#rawJson"),
  replyAudio: document.querySelector("#replyAudio"),
  recordedAudio: document.querySelector("#recordedAudio"),
  recordingHint: document.querySelector("#recordingHint"),
};

function renderSongs() {
  els.songGrid.innerHTML = "";
  for (const song of state.songs) {
    const button = document.createElement("button");
    button.className = `song${song.id === state.selected.id && song.lineId === state.selected.lineId ? " selected" : ""}`;
    button.type = "button";
    button.innerHTML = `<strong>${song.name}</strong><span>${song.text || song.id}</span>`;
    button.addEventListener("click", () => {
      state.selected = song;
      renderSongs();
    });
    els.songGrid.appendChild(button);
  }
}

function setStatus(label, mode = "") {
  els.connectionStatus.textContent = label;
  els.connectionStatus.className = `status ${mode}`.trim();
}

function clearResult() {
  els.resultTitle.textContent = "等待测试";
  els.endToResult.textContent = "--";
  els.matched.textContent = "--";
  els.songId.textContent = "--";
  els.confidence.textContent = "--";
  els.matchedLine.textContent = "--";
  els.rawJson.textContent = "{}";
  els.replyAudio.removeAttribute("src");
  els.replyAudio.load();
  setStatus("Idle");
}

function updateResult(result) {
  els.resultTitle.textContent = result.matched ? result.song_name || "Matched" : "未匹配";
  els.endToResult.textContent = `${result.latency_ms?.end_to_result ?? "--"}ms`;
  els.matched.textContent = String(result.matched);
  els.songId.textContent = result.song_id || "--";
  els.confidence.textContent = typeof result.confidence === "number" ? result.confidence.toFixed(4) : "--";
  els.matchedLine.textContent = result.matched_line || "--";
  els.rawJson.textContent = JSON.stringify(result, null, 2);

  if (result.reply_audio_url) {
    els.replyAudio.src = result.reply_audio_url;
    els.replyAudio.load();
  }
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function floatToPcm16(float32) {
  const buffer = new ArrayBuffer(float32.length * 2);
  const view = new DataView(buffer);
  for (let index = 0; index < float32.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, float32[index]));
    view.setInt16(index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}

function midiToHz(midi) {
  return 440 * 2 ** ((midi - 69) / 12);
}

function synthMelodyPcmChunks(features, sampleRate = 16000, noteMs = 320) {
  const chunks = [];
  const samplesPerNote = Math.floor((sampleRate * noteMs) / 1000);
  const amplitude = 0.32;

  for (const midi of features) {
    const freq = midiToHz(midi);
    const float32 = new Float32Array(samplesPerNote);
    for (let index = 0; index < samplesPerNote; index += 1) {
      const fadeIn = Math.min(1, index / Math.max(1, sampleRate * 0.025));
      const fadeOut = Math.min(1, (samplesPerNote - index) / Math.max(1, sampleRate * 0.035));
      const envelope = Math.min(fadeIn, fadeOut);
      float32[index] = amplitude * envelope * Math.sin((2 * Math.PI * freq * index) / sampleRate);
    }
    chunks.push(floatToPcm16(float32));
  }

  return chunks;
}

function buildWavBlob(pcmChunks, sampleRate) {
  const dataLength = pcmChunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  const wav = new ArrayBuffer(44 + dataLength);
  const view = new DataView(wav);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (const chunk of pcmChunks) {
    new Uint8Array(wav, offset, chunk.byteLength).set(new Uint8Array(chunk));
    offset += chunk.byteLength;
  }
  return new Blob([wav], { type: "audio/wav" });
}

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function setRecordedAudio(pcmChunks, sampleRate) {
  if (state.recordedUrl) {
    URL.revokeObjectURL(state.recordedUrl);
  }
  if (!pcmChunks.length) {
    els.recordedAudio.removeAttribute("src");
    els.recordedAudio.load();
    els.recordingHint.textContent = "没有录到可回放音频。";
    return;
  }
  const blob = buildWavBlob(pcmChunks, sampleRate);
  state.recordedUrl = URL.createObjectURL(blob);
  els.recordedAudio.src = state.recordedUrl;
  els.recordedAudio.load();
  els.recordingHint.textContent = `已保存最近一次录音，约 ${(blob.size / 1024).toFixed(1)} KB。`;
}

function setRecordedAudioUrl(url, hint) {
  if (state.recordedUrl) {
    URL.revokeObjectURL(state.recordedUrl);
    state.recordedUrl = null;
  }
  els.recordedAudio.src = url;
  els.recordedAudio.load();
  els.recordingHint.textContent = hint;
}

async function waitForOpen(socket) {
  return new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
}

async function waitForResult(socket, endSentAt, started) {
  return new Promise((resolve, reject) => {
    socket.addEventListener("message", (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "result") {
        payload.client_latency_ms = {
          end_to_result: Math.round(performance.now() - endSentAt),
          total: Math.round(performance.now() - started),
        };
        resolve(payload);
      }
    });
    socket.addEventListener("error", reject, { once: true });
    socket.addEventListener("close", () => reject(new Error("socket closed before result")), { once: true });
  });
}

async function sendDemo() {
  if (state.running) return;
  state.running = true;
  els.sendButton.disabled = true;
  els.randomPromptButton.disabled = true;
  els.clearButton.disabled = true;
  clearResult();
  setStatus("Connecting", "active");

  const socket = new WebSocket(els.wsUrl.value.trim());
  const started = performance.now();

  try {
    await waitForOpen(socket);
    setStatus("Streaming", "active");

    socket.send(
      JSON.stringify({
        type: "start",
        catalog_id: els.catalogId.value.trim() || "mao_buyi_v1",
        sample_rate: 16000,
        format: "pcm_s16le",
      })
    );

    for (let offset = 0; offset < state.selected.features.length; offset += 3) {
      const chunk = state.selected.features.slice(offset, offset + 3);
      socket.send(JSON.stringify({ type: "demo_features", values: chunk }));
      await sleep(50);
    }

    const endSentAt = performance.now();
    socket.send(JSON.stringify({ type: "end" }));
    const result = await waitForResult(socket, endSentAt, started);
    updateResult(result);
    setStatus("Done", "active");
  } catch (error) {
    setStatus("Error", "error");
    els.resultTitle.textContent = "连接失败";
    els.rawJson.textContent = String(error?.message || error);
  } finally {
    if (socket.readyState === WebSocket.OPEN) {
      socket.close();
    }
    state.running = false;
    els.sendButton.disabled = false;
    els.randomPromptButton.disabled = false;
    els.clearButton.disabled = false;
  }
}

async function loadCatalog() {
  const catalogId = els.catalogId.value.trim() || "mao_buyi_v1";
  try {
    const response = await fetch(`/v1/catalog/${encodeURIComponent(catalogId)}`);
    if (!response.ok) throw new Error(`catalog HTTP ${response.status}`);
    const payload = await response.json();
    state.songs = payload.songs.map((item) => ({
      id: item.song_id,
      name: item.song_name,
      lineId: item.line_id,
      text: item.text,
      promptAudioUrl: item.prompt_audio_url,
      features: item.features,
    }));
    state.selected = state.songs[0] || fallbackSongs[0];
    renderSongs();
  } catch (error) {
    state.songs = fallbackSongs;
    state.selected = fallbackSongs[0];
    renderSongs();
    els.recordingHint.textContent = `曲库加载失败，使用 fallback demo：${error.message}`;
  }
}

async function playRandomPrompt() {
  if (state.running) return;
  const song = state.songs[Math.floor(Math.random() * state.songs.length)];
  state.selected = song;
  renderSongs();
  setStatus("Prompt");

  let usedOriginal = false;
  try {
    const response = await fetch(song.promptAudioUrl, { method: "HEAD" });
    usedOriginal = response.ok;
  } catch (_) {
    usedOriginal = false;
  }

  if (usedOriginal) {
    setRecordedAudioUrl(song.promptAudioUrl, `随机原唱片段：${song.name} / ${song.text}。听完后可以点“开始录音”接唱。`);
  } else {
    const sampleRate = 16000;
    const pcmChunks = synthMelodyPcmChunks(song.features, sampleRate);
    setRecordedAudio(pcmChunks, sampleRate);
    els.recordingHint.textContent = `未找到授权原唱片段，已播放合成提示：${song.name} / ${song.text}。`;
  }

  try {
    await els.recordedAudio.play();
  } catch (_) {
    // Some browsers still block autoplay; the user can press the audio control.
  }
}

async function startRecording() {
  if (state.running || state.recorder) return;
  state.running = true;
  els.sendButton.disabled = true;
  els.randomPromptButton.disabled = true;
  els.clearButton.disabled = true;
  els.recordButton.textContent = "停止并识别";
  clearResult();
  setStatus("Connecting", "active");

  const socket = new WebSocket(els.wsUrl.value.trim());
  const started = performance.now();
  let audioContext = null;
  let mediaStream = null;
  let processor = null;
  let source = null;
  const pcmChunks = [];

  try {
    await waitForOpen(socket);

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });

    audioContext = new AudioContext({ sampleRate: 16000 });
    source = audioContext.createMediaStreamSource(mediaStream);
    processor = audioContext.createScriptProcessor(2048, 1, 1);

    socket.send(
      JSON.stringify({
        type: "start",
        catalog_id: els.catalogId.value.trim() || "mao_buyi_v1",
        sample_rate: audioContext.sampleRate,
        format: "pcm_s16le",
      })
    );

    processor.onaudioprocess = (event) => {
      if (socket.readyState !== WebSocket.OPEN) return;
      const input = event.inputBuffer.getChannelData(0);
      const pcm = floatToPcm16(input);
      pcmChunks.push(pcm.slice(0));
      socket.send(pcm);
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
    setStatus("Recording", "active");

    state.recorder = {
      socket,
      audioContext,
      mediaStream,
      processor,
      source,
      started,
      pcmChunks,
      sampleRate: audioContext.sampleRate,
    };
  } catch (error) {
    if (processor) processor.disconnect();
    if (source) source.disconnect();
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
    }
    if (audioContext) {
      await audioContext.close();
    }
    if (socket.readyState === WebSocket.OPEN) {
      socket.close();
    }
    state.running = false;
    state.recorder = null;
    els.sendButton.disabled = false;
    els.randomPromptButton.disabled = false;
    els.clearButton.disabled = false;
    els.recordButton.textContent = "开始录音";
    setStatus("Error", "error");
    els.resultTitle.textContent = "录音失败";
    els.rawJson.textContent = String(error?.message || error);
  }
}

async function stopRecording() {
  const recorder = state.recorder;
  if (!recorder) return;

  state.recorder = null;
  setStatus("Finalizing", "active");
  els.recordButton.disabled = true;

  try {
    recorder.processor.disconnect();
    recorder.source.disconnect();
    recorder.mediaStream.getTracks().forEach((track) => track.stop());
    await recorder.audioContext.close();
    setRecordedAudio(recorder.pcmChunks, recorder.sampleRate);

    const endSentAt = performance.now();
    recorder.socket.send(JSON.stringify({ type: "end" }));
    const result = await waitForResult(recorder.socket, endSentAt, recorder.started);
    updateResult(result);
    setStatus("Done", "active");
  } catch (error) {
    setStatus("Error", "error");
    els.resultTitle.textContent = "识别失败";
    els.rawJson.textContent = String(error?.message || error);
  } finally {
    if (recorder.socket.readyState === WebSocket.OPEN) {
      recorder.socket.close();
    }
    state.running = false;
    els.sendButton.disabled = false;
    els.randomPromptButton.disabled = false;
    els.clearButton.disabled = false;
    els.recordButton.disabled = false;
    els.recordButton.textContent = "开始录音";
  }
}

async function toggleRecording() {
  if (state.recorder) {
    await stopRecording();
  } else {
    await startRecording();
  }
}

els.sendButton.addEventListener("click", sendDemo);
els.randomPromptButton.addEventListener("click", playRandomPrompt);
els.recordButton.addEventListener("click", toggleRecording);
els.clearButton.addEventListener("click", clearResult);
els.catalogId.addEventListener("change", loadCatalog);

clearResult();
loadCatalog();

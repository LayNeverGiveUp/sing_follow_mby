const state = {
  running: false,
  recorder: null,
  testItems: null,
  remainingTestItems: [],
  selectedTest: null,
  recordedAudioUrl: null,
  recordingStartedAt: null,
  recordingTimer: null,
};

const els = {
  vocalTestButton: document.querySelector("#vocalTestButton"),
  recordButton: document.querySelector("#recordButton"),
  clearButton: document.querySelector("#clearButton"),
  wsUrl: document.querySelector("#wsUrl"),
  connectionStatus: document.querySelector("#connectionStatus"),
  resultTitle: document.querySelector("#resultTitle"),
  endToResult: document.querySelector("#endToResult"),
  matched: document.querySelector("#matched"),
  songId: document.querySelector("#songId"),
  confidence: document.querySelector("#confidence"),
  matchedLine: document.querySelector("#matchedLine"),
  rawJson: document.querySelector("#rawJson"),
  queryAudio: document.querySelector("#queryAudio"),
  nextAudio: document.querySelector("#nextAudio"),
  queryLabel: document.querySelector("#queryLabel"),
  nextLabel: document.querySelector("#nextLabel"),
  recordedAudio: document.querySelector("#recordedAudio"),
  recordingHint: document.querySelector("#recordingHint"),
  eventLog: document.querySelector("#eventLog"),
};

function logEvent(message, detail) {
  const timestamp = new Date().toLocaleTimeString();
  const suffix = detail === undefined ? "" : ` ${typeof detail === "string" ? detail : JSON.stringify(detail)}`;
  const lines = els.eventLog.textContent === "等待操作。" ? [] : els.eventLog.textContent.split("\n");
  lines.push(`[${timestamp}] ${message}${suffix}`);
  els.eventLog.textContent = lines.slice(-12).join("\n");
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
  els.eventLog.textContent = "等待操作。";
  setStatus("Idle");
}

function clearRecordedAudio() {
  if (state.recordedAudioUrl) URL.revokeObjectURL(state.recordedAudioUrl);
  state.recordedAudioUrl = null;
  els.recordedAudio.removeAttribute("src");
  els.recordedAudio.load();
}

function updateResult(result) {
  const accepted = Boolean(result.accepted ?? result.matched);
  els.resultTitle.textContent = accepted ? result.song_name || result.song_id || "已匹配" : "未匹配";
  els.endToResult.textContent = `${result.latency_ms?.end_to_result ?? "--"}ms`;
  els.matched.textContent = String(accepted);
  els.songId.textContent = result.song_id || "--";
  const score = result.score ?? result.confidence;
  els.confidence.textContent = typeof score === "number" ? score.toFixed(4) : "--";
  const current = result.current_lyric_text || result.matched_line || "--";
  const next = result.next_lyric_text ? ` → ${result.next_lyric_text}` : "";
  els.matchedLine.textContent = `${current}${next}`;
  els.rawJson.textContent = JSON.stringify(result, null, 2);
}

function setButtonsDisabled(disabled) {
  els.vocalTestButton.disabled = disabled;
  els.clearButton.disabled = disabled;
  if (!state.recorder) els.recordButton.disabled = disabled;
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

function pcmChunksToWav(chunks, sampleRate) {
  const byteLength = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  const wav = new ArrayBuffer(44 + byteLength);
  const view = new DataView(wav);
  const write = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  write(0, "RIFF");
  view.setUint32(4, 36 + byteLength, true);
  write(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, byteLength, true);
  let offset = 44;
  for (const chunk of chunks) {
    new Uint8Array(wav, offset, chunk.byteLength).set(new Uint8Array(chunk));
    offset += chunk.byteLength;
  }
  return new Blob([wav], { type: "audio/wav" });
}

function stopRecordingTimer() {
  if (state.recordingTimer) window.clearInterval(state.recordingTimer);
  state.recordingTimer = null;
}

function waitForOpen(socket) {
  return new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", () => reject(new Error("WebSocket 连接失败")), { once: true });
  });
}

function waitForResult(socket, endSentAt) {
  return new Promise((resolve, reject) => {
    socket.addEventListener("message", (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "ack") {
        logEvent("服务已接收录音流", payload);
        return;
      }
      if (payload.type === "error") reject(new Error(payload.message));
      if (payload.type === "result") {
        payload.client_latency_ms = Math.round(performance.now() - endSentAt);
        resolve(payload);
      }
    });
    socket.addEventListener("error", () => reject(new Error("WebSocket 通信失败")), { once: true });
    socket.addEventListener("close", () => reject(new Error("服务在返回结果前关闭连接")), { once: true });
  });
}

function audioContext() {
  const Context = window.AudioContext || window.webkitAudioContext;
  if (!Context) throw new Error("当前浏览器不支持 Web Audio API");
  return new Context({ sampleRate: 16000 });
}

async function selectRandomTest() {
  if (!state.testItems) {
    const response = await fetch("/v1/hum-mvp/test-queries", { cache: "no-store" });
    if (!response.ok) throw new Error("测试句库加载失败");
    const payload = await response.json();
    state.testItems = payload.items || [];
  }
  if (!state.testItems.length) throw new Error("没有可用的带下一句测试素材");
  if (!state.remainingTestItems.length) {
    state.remainingTestItems = shuffle(state.testItems);
  }
  state.selectedTest = state.remainingTestItems.pop();
  const item = state.selectedTest;
  els.queryAudio.src = item.query_audio_url;
  els.queryAudio.load();
  els.queryLabel.textContent = `抽中输入句：${item.song_id}｜${item.current_lyric_text}`;
  els.nextAudio.removeAttribute("src");
  els.nextAudio.load();
  els.nextLabel.textContent = "等待识别后确定下一句…";
  return item;
}

function shuffle(items) {
  const shuffled = [...items];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const random = new Uint32Array(1);
    crypto.getRandomValues(random);
    const swapIndex = random[0] % (index + 1);
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled;
}

function showRecognitionAudio(result) {
  if (!result.accepted || !result.song_id || !Number.isInteger(result.current_lyric_index)) {
    els.nextAudio.removeAttribute("src");
    els.nextAudio.load();
    els.nextLabel.textContent = "未识别成功，因此不播放下一句。";
    return;
  }
  const song = encodeURIComponent(result.song_id);
  const lineUrl = (index) => `/static/queries/mao_buyi_v1/${song}/line_${String(index).padStart(3, "0")}.wav`;
  // The upper player remains the randomly selected input clip. Recognition
  // results only determine the next-line player and the result panel.
  if (Number.isInteger(result.next_lyric_index)) {
    els.nextAudio.src = lineUrl(result.next_lyric_index);
    els.nextAudio.load();
    els.nextLabel.textContent = `识别下一句：${result.next_lyric_text || "--"}`;
  } else {
    els.nextAudio.removeAttribute("src");
    els.nextAudio.load();
    els.nextLabel.textContent = "识别结果已是最后一句，没有下一句。";
  }
}

async function sendPcmForMvp(pcmChunks, sampleRate) {
  const socket = new WebSocket(els.wsUrl.value.trim());
  await waitForOpen(socket);
  socket.send(JSON.stringify({ type: "start", catalog_id: "mao_buyi_v1", matcher_mode: "hum_song_mvp", sample_rate: sampleRate, format: "pcm_s16le" }));
  for (const chunk of pcmChunks) socket.send(chunk);
  logEvent("已发送 PCM", { sample_rate: sampleRate, chunks: pcmChunks.length, bytes: pcmChunks.reduce((total, chunk) => total + chunk.byteLength, 0) });
  const endSentAt = performance.now();
  socket.send(JSON.stringify({ type: "end" }));
  try {
    return await waitForResult(socket, endSentAt);
  } finally {
    if (socket.readyState === WebSocket.OPEN) socket.close();
  }
}

async function sendVocalTest() {
  if (state.running) return;
  state.running = true;
  setButtonsDisabled(true);
  clearResult();
  let context;
  try {
    setStatus("Loading test vocal", "active");
    const selected = await selectRandomTest();
    els.queryLabel.textContent = `抽中输入句：${selected.song_id}｜${selected.current_lyric_text}（正在识别…）`;
    const response = await fetch(selected.query_audio_url, { cache: "no-store" });
    if (!response.ok) throw new Error("未找到干声测试素材");
    context = audioContext();
    const decoded = await context.decodeAudioData(await response.arrayBuffer());
    const mono = decoded.getChannelData(0);
    const chunks = [];
    for (let offset = 0; offset < mono.length; offset += 2048) chunks.push(floatToPcm16(mono.subarray(offset, offset + 2048)));
    setStatus("Matching", "active");
    const result = await sendPcmForMvp(chunks, context.sampleRate);
    logEvent("识别结果", { accepted: result.accepted, reason: result.reason, diagnostics: result.diagnostics });
    updateResult(result);
    showRecognitionAudio(result);
    setStatus("Done", "active");
  } catch (error) {
    els.resultTitle.textContent = "测试失败";
    els.rawJson.textContent = String(error?.message || error);
    logEvent("干声测试失败", String(error?.message || error));
    els.queryLabel.textContent = "干声测试未能完成，请查看下方错误信息。";
    setStatus("Error", "error");
  } finally {
    if (context) await context.close();
    state.running = false;
    setButtonsDisabled(false);
  }
}

async function startRecording() {
  if (state.running || state.recorder) return;
  state.running = true;
  setButtonsDisabled(true);
  els.recordButton.disabled = false;
  els.recordButton.textContent = "停止并识别";
  clearResult();
  clearRecordedAudio();
  let context;
  let mediaStream;
  let processor;
  let source;
  try {
    // In normal office environments, browser voice processing produces a more
    // usable recording than an unprocessed microphone signal.
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    context = audioContext();
    source = context.createMediaStreamSource(mediaStream);
    processor = context.createScriptProcessor(2048, 1, 1);
    const pcmChunks = [];
    processor.onaudioprocess = (event) => pcmChunks.push(floatToPcm16(event.inputBuffer.getChannelData(0)).slice(0));
    source.connect(processor);
    // Keep ScriptProcessor alive without routing the microphone back to speakers.
    const silentGain = context.createGain();
    silentGain.gain.value = 0;
    processor.connect(silentGain);
    silentGain.connect(context.destination);
    state.recorder = { context, mediaStream, processor, source, silentGain, pcmChunks };
    state.recordingStartedAt = performance.now();
    state.recordingTimer = window.setInterval(() => {
      const seconds = ((performance.now() - state.recordingStartedAt) / 1000).toFixed(1);
      els.recordingHint.textContent = `正在录音 ${seconds} 秒；建议录满 4–8 秒后点击“停止并识别”。`;
    }, 100);
    els.recordingHint.textContent = "正在请求麦克风…";
    logEvent("开始录音", { sample_rate: context.sampleRate });
    setStatus("Recording", "active");
  } catch (error) {
    if (processor) processor.disconnect();
    if (source) source.disconnect();
    if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
    if (context) await context.close();
    state.running = false;
    setButtonsDisabled(false);
    els.recordButton.textContent = "开始录音";
    els.recordingHint.textContent = "无法启动录音；请允许浏览器使用麦克风。";
    logEvent("无法启动录音", String(error?.message || error));
    els.resultTitle.textContent = "录音失败";
    els.rawJson.textContent = String(error?.message || error);
    setStatus("Error", "error");
  }
}

async function stopRecording() {
  const recorder = state.recorder;
  if (!recorder) return;
  state.recorder = null;
  stopRecordingTimer();
  els.recordButton.disabled = true;
  try {
    recorder.processor.disconnect();
    recorder.source.disconnect();
    recorder.silentGain.disconnect();
    recorder.mediaStream.getTracks().forEach((track) => track.stop());
    await recorder.context.close();
    state.recordedAudioUrl = URL.createObjectURL(pcmChunksToWav(recorder.pcmChunks, recorder.context.sampleRate));
    els.recordedAudio.src = state.recordedAudioUrl;
    els.recordedAudio.load();
    const recordedSeconds = ((performance.now() - state.recordingStartedAt) / 1000).toFixed(1);
    els.recordingHint.textContent = `已录制 ${recordedSeconds} 秒，正在识别…`;
    logEvent("停止录音", { duration_seconds: Number(recordedSeconds), chunks: recorder.pcmChunks.length });
    setStatus("Matching", "active");
    const result = await sendPcmForMvp(recorder.pcmChunks, recorder.context.sampleRate);
    logEvent("识别结果", { accepted: result.accepted, reason: result.reason, diagnostics: result.diagnostics });
    updateResult(result);
    showRecognitionAudio(result);
    els.recordingHint.textContent = `已录制 ${recordedSeconds} 秒，识别完成。`;
    setStatus("Done", "active");
  } catch (error) {
    els.resultTitle.textContent = "识别失败";
    els.rawJson.textContent = String(error?.message || error);
    logEvent("录音识别失败", String(error?.message || error));
    els.recordingHint.textContent = "录音已保留，但识别失败；可播放检查后重试。";
    setStatus("Error", "error");
  } finally {
    state.running = false;
    setButtonsDisabled(false);
    els.recordButton.disabled = false;
    els.recordButton.textContent = "开始录音";
    state.recordingStartedAt = null;
  }
}

els.vocalTestButton.addEventListener("click", sendVocalTest);
els.recordButton.addEventListener("click", () => (state.recorder ? stopRecording() : startRecording()));
els.clearButton.addEventListener("click", clearResult);
clearResult();

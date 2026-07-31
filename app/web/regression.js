const state = {
  plan: null,
  index: 0,
  recorder: null,
  recordedAudioUrl: null,
  busy: false,
};

const els = {
  progressText: document.querySelector("#progressText"),
  progressBar: document.querySelector("#progressBar"),
  caseMeta: document.querySelector("#caseMeta"),
  lyricText: document.querySelector("#lyricText"),
  nextLyricText: document.querySelector("#nextLyricText"),
  purposeText: document.querySelector("#purposeText"),
  referenceAudio: document.querySelector("#referenceAudio"),
  recordedAudio: document.querySelector("#recordedAudio"),
  previousButton: document.querySelector("#previousButton"),
  recordButton: document.querySelector("#recordButton"),
  nextButton: document.querySelector("#nextButton"),
  recordingHint: document.querySelector("#recordingHint"),
  resultTitle: document.querySelector("#resultTitle"),
  saveStatus: document.querySelector("#saveStatus"),
  resultJson: document.querySelector("#resultJson"),
  caseList: document.querySelector("#caseList"),
};

const websocketProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const websocketUrl = `${websocketProtocol}//${window.location.host}/v1/realtime-match`;

function currentCase() {
  return state.plan?.items[state.index] || null;
}

function setBusy(busy) {
  state.busy = busy;
  els.recordButton.disabled = busy && !state.recorder;
  els.previousButton.disabled = busy || state.index === 0;
  els.nextButton.disabled = busy || !state.plan || state.index === state.plan.items.length - 1;
  els.caseList.querySelectorAll("button").forEach((button) => {
    button.disabled = busy;
  });
}

function clearRecordedAudio() {
  if (state.recordedAudioUrl) URL.revokeObjectURL(state.recordedAudioUrl);
  state.recordedAudioUrl = null;
  els.recordedAudio.removeAttribute("src");
  els.recordedAudio.load();
}

function render() {
  if (!state.plan) return;
  const item = currentCase();
  const recordedCount = state.plan.items.filter((entry) => entry.recorded).length;
  els.progressText.textContent = `已保存 ${recordedCount} / ${state.plan.total_count} 句`;
  els.progressBar.style.width = `${(recordedCount / state.plan.total_count) * 100}%`;
  els.caseMeta.textContent = `第 ${item.ordinal} / ${state.plan.total_count} 句｜${item.song_id}｜歌词行 ${item.lyric_index}`;
  els.lyricText.textContent = item.lyric_text;
  els.nextLyricText.textContent = item.next_lyric_text;
  els.purposeText.textContent = `测试目的：${item.purpose}`;
  els.referenceAudio.src = item.reference_audio_url;
  els.referenceAudio.load();
  els.recordButton.textContent = state.recorder
    ? "停止录制并识别"
    : item.recorded
      ? "重新录制这一句"
      : "开始录制";
  els.previousButton.disabled = state.busy || state.index === 0;
  els.nextButton.disabled = state.busy || state.index === state.plan.items.length - 1;

  els.caseList.innerHTML = "";
  state.plan.items.forEach((entry, index) => {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.className = `case-button${index === state.index ? " current" : ""}${entry.recorded ? " recorded" : ""}`;
    button.disabled = state.busy;
    const label = document.createTextNode(`${entry.recorded ? "✓" : entry.ordinal}. ${entry.lyric_text}`);
    const detail = document.createElement("small");
    detail.textContent = `${entry.song_id}｜${entry.purpose}`;
    button.append(label, detail);
    button.addEventListener("click", () => selectCase(index));
    li.appendChild(button);
    els.caseList.appendChild(li);
  });
}

function selectCase(index) {
  if (state.busy || state.recorder || !state.plan) return;
  state.index = Math.max(0, Math.min(index, state.plan.items.length - 1));
  clearRecordedAudio();
  els.resultTitle.textContent = currentCase().recorded ? "该句已有录音，可重新录制" : "等待录制";
  els.saveStatus.textContent = currentCase().recorded ? "已保存" : "";
  els.resultJson.textContent = "{}";
  els.recordingHint.textContent = "先听参考切片熟悉内容，再点击“开始录制”，完整唱完当前一句后停止。";
  render();
}

function audioContext() {
  const Context = window.AudioContext || window.webkitAudioContext;
  if (!Context) throw new Error("当前浏览器不支持 Web Audio API");
  return new Context({ sampleRate: 16000 });
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

function waitForOpen(socket) {
  return new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", () => reject(new Error("WebSocket 连接失败")), { once: true });
  });
}

function waitForResult(socket) {
  return new Promise((resolve, reject) => {
    socket.addEventListener("message", (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "error") reject(new Error(payload.message));
      if (payload.type === "result") resolve(payload);
    });
    socket.addEventListener("error", () => reject(new Error("WebSocket 通信失败")), { once: true });
  });
}

async function sendRecording(item, chunks, sampleRate) {
  const socket = new WebSocket(websocketUrl);
  await waitForOpen(socket);
  socket.send(JSON.stringify({
    type: "start",
    catalog_id: "mao_buyi_v1",
    matcher_mode: "hum_song_mvp",
    sample_rate: sampleRate,
    format: "pcm_s16le",
    input_source: "manual_regression",
    test_plan_id: state.plan.plan_id,
    test_case_id: item.case_id,
    expected_song_id: item.song_id,
    expected_lyric_index: item.lyric_index,
  }));
  for (const chunk of chunks) socket.send(chunk);
  socket.send(JSON.stringify({ type: "end" }));
  try {
    return await waitForResult(socket);
  } finally {
    if (socket.readyState === WebSocket.OPEN) socket.close();
  }
}

async function startRecording() {
  if (state.busy || state.recorder) return;
  setBusy(true);
  clearRecordedAudio();
  let context;
  let stream;
  let source;
  let processor;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    context = audioContext();
    source = context.createMediaStreamSource(stream);
    processor = context.createScriptProcessor(2048, 1, 1);
    const chunks = [];
    processor.onaudioprocess = (event) => {
      chunks.push(floatToPcm16(event.inputBuffer.getChannelData(0)).slice(0));
    };
    const silentGain = context.createGain();
    silentGain.gain.value = 0;
    source.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(context.destination);
    state.recorder = {
      context,
      stream,
      source,
      processor,
      silentGain,
      chunks,
      startedAt: performance.now(),
      timer: window.setInterval(() => {
        const seconds = ((performance.now() - state.recorder.startedAt) / 1000).toFixed(1);
        els.recordingHint.textContent = `正在录制 ${seconds} 秒；唱完当前整句后点击停止。`;
      }, 100),
    };
    els.recordButton.disabled = false;
    els.resultTitle.textContent = "正在录制";
    els.saveStatus.textContent = "";
    render();
  } catch (error) {
    if (processor) processor.disconnect();
    if (source) source.disconnect();
    if (stream) stream.getTracks().forEach((track) => track.stop());
    if (context) await context.close();
    state.recorder = null;
    setBusy(false);
    els.resultTitle.textContent = "无法启动录音";
    els.recordingHint.textContent = "请允许浏览器使用麦克风后重试。";
    els.resultJson.textContent = String(error?.message || error);
  }
}

async function stopRecording() {
  const recorder = state.recorder;
  if (!recorder) return;
  const item = currentCase();
  state.recorder = null;
  window.clearInterval(recorder.timer);
  els.recordButton.disabled = true;
  try {
    recorder.processor.disconnect();
    recorder.source.disconnect();
    recorder.silentGain.disconnect();
    recorder.stream.getTracks().forEach((track) => track.stop());
    const sampleRate = recorder.context.sampleRate;
    await recorder.context.close();
    const duration = (performance.now() - recorder.startedAt) / 1000;
    state.recordedAudioUrl = URL.createObjectURL(pcmChunksToWav(recorder.chunks, sampleRate));
    els.recordedAudio.src = state.recordedAudioUrl;
    els.recordedAudio.load();
    els.recordingHint.textContent = `已录制 ${duration.toFixed(1)} 秒，正在识别并保存测试样本…`;
    els.resultTitle.textContent = "正在识别";
    const result = await sendRecording(item, recorder.chunks, sampleRate);
    item.recorded = Boolean(result.test_dataset_saved);
    els.resultTitle.textContent = result.accepted
      ? `识别成功：${result.song_id}`
      : `已保存，识别结果：${result.reason || "未匹配"}`;
    els.saveStatus.textContent = result.test_dataset_saved ? "测试样本已保存" : "保存失败";
    els.resultJson.textContent = JSON.stringify(result, null, 2);
    els.recordingHint.textContent = result.test_dataset_saved
      ? "录音、期望标签和完整识别日志均已保存。可以进入下一句。"
      : "识别已完成，但测试集保存失败，请查看结果。";
  } catch (error) {
    els.resultTitle.textContent = "录音处理失败";
    els.saveStatus.textContent = "";
    els.resultJson.textContent = String(error?.message || error);
    els.recordingHint.textContent = "本句未保存，请重试。";
  } finally {
    setBusy(false);
    render();
  }
}

async function loadPlan() {
  const response = await fetch("/v1/hum-mvp/regression-plan", { cache: "no-store" });
  if (!response.ok) throw new Error(`录制计划加载失败：HTTP ${response.status}`);
  state.plan = await response.json();
  const firstPending = state.plan.items.findIndex((item) => !item.recorded);
  state.index = firstPending >= 0 ? firstPending : 0;
  render();
}

els.previousButton.addEventListener("click", () => selectCase(state.index - 1));
els.nextButton.addEventListener("click", () => selectCase(state.index + 1));
els.recordButton.addEventListener("click", () => (state.recorder ? stopRecording() : startRecording()));

loadPlan().catch((error) => {
  els.resultTitle.textContent = "页面初始化失败";
  els.resultJson.textContent = String(error?.message || error);
  els.recordButton.disabled = true;
});

// Dependency-free tests of the actual frontend script, with controlled browser APIs.
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { randomUUID } = require('node:crypto');
const html = fs.readFileSync(path.resolve(__dirname, '../../webdemo/static/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
async function settle() { for (let i = 0; i < 15; i++) await Promise.resolve(); }
function response(data, status = 200) { return { ok: status < 400, status, json: async () => data }; }

function browser(storage = new Map()) {
  class Element {
    constructor() {
      this.children = []; this.style = {}; this.textContent = ''; this.files = [];
      this.dataset = {}; this.events = {}; this.open = false;
      this.classList = { toggle() {}, add() {}, remove() {} };
    }
    set innerHTML(value) { throw new Error('HTML rendering is forbidden: ' + value); }
    querySelector(selector) { return this.children.find(child => '.' + child.className === selector) || null; }
    replaceChildren(...children) { this.children = children; }
    appendChild(child) { this.children.push(child); }
    addEventListener(type, handler) { this.events[type] = handler; }
    click() { return this.events.click?.(); }
    showModal() { this.open = true; }
    close() { this.open = false; }
  }
  const elements = new Map(), requests = [], contexts = [], streams = [], timers = new Map();
  let now = 1000, timerID = 0, sessionNumber = 0;
  const get = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  class AudioContext {
    constructor() { this.sampleRate = 48000; this.closed = false; contexts.push(this); }
    async resume() {}
    async close() { this.closed = true; }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createScriptProcessor() { return { connect() {}, disconnect() {} }; }
  }
  const sandbox = {
    console, FormData, Blob, AbortController, crypto: { randomUUID },
    Date: class extends Date { static now() { return now; } },
    setTimeout(fn, ms) { const id = ++timerID; timers.set(id, { fn, due: now + ms }); return id; },
    clearTimeout(id) { timers.delete(id); },
    document: { documentElement: new Element(), getElementById: get, createElement: () => new Element(), querySelectorAll: () => [] },
    localStorage: { getItem(key) { return storage.get(key) ?? null; }, setItem(key, value) { storage.set(key, value); }, removeItem(key) { storage.delete(key); } },
    window: {
      SpeechRecognition: class { start() {} abort() {} },
      AudioContext, addEventListener() {}, confirm() { return true; },
    },
    navigator: { mediaDevices: { async getUserMedia() {
      const track = { stopped: false, stop() { this.stopped = true; } };
      const stream = { getTracks: () => [track], track }; streams.push(stream); return stream;
    } } },
    async fetch(route, options = {}) {
      requests.push({ route, options });
      if (sandbox.handleFetch) return sandbox.handleFetch(route, options);
      if (route === '/api/live/start') return response({ session_id: 'session-' + ++sessionNumber });
      return response({ ended: true, risk_score: 0.4, action: 'warn', scam_score: 0.4, spoof_score: 0 });
    },
  };
  const context = vm.createContext(sandbox);
  vm.runInContext(script, context);
  return { sandbox, get, requests, streams, contexts,
    run: code => vm.runInContext(code, context),
    async advance(ms) {
      now += ms;
      const ready = [...timers].filter(([, timer]) => timer.due <= now);
      for (const [id, timer] of ready) { timers.delete(id); timer.fn(); }
      await settle();
    },
  };
}

test('untrusted explanation and error content is rendered only as text', () => {
  const b = browser();
  b.run(`appendLogEntry(document.getElementById('log'), {time:'<script>bad()</script>',action:'warn',text:'<img src=x onerror=bad()>'}, 0)`);
  assert.equal(b.get('log').children[0].children[2].textContent, '<img src=x onerror=bad()>');
  b.run(`showEmpty(document.getElementById('log'), '<svg onload=bad()>')`);
  assert.equal(b.get('log').children[0].textContent, '<svg onload=bad()>');
});

test('microphone permission denial leaves no session and restores controls', async () => {
  const b = browser();
  b.sandbox.navigator.mediaDevices.getUserMedia = async () => { throw new Error('permission denied'); };
  await b.run('startLiveCall()');
  assert.equal(b.requests.length, 0);
  assert.equal(b.run('currentCall'), null);
  assert.equal(b.get('startBtn').style.display, 'inline-block');
  assert.match(b.get('micStatus').textContent, /permission denied/);
});

test('full session transcripts persist and reload as plain text', () => {
  const storage = new Map(), b = browser(storage);
  const transcript = '<img src=x onerror=bad()> नमस्ते बैंक\n'.repeat(100) + 'FINAL WORDS';
  b.sandbox.fixture = {id:'fixture',timestamp:1000,mode:'transcript',transcript,risk:.4,action:'warn',unconfirmed:false};
  b.run('saveHistory(fixture); openHistory("fixture")');
  assert.equal(b.get('historyTranscript').textContent, transcript);
  assert.equal(b.get('historyDialog').open, true);
  assert.equal(b.get('historyList').children.length, 1);
  const restored = browser(storage);
  restored.run('openHistory("fixture")');
  assert.equal(restored.get('historyTranscript').textContent, transcript);
  restored.get('closeHistoryDialog').click();
  assert.equal(restored.get('historyDialog').open, false);
});

test('successful paste analysis saves the analyzed transcript rather than later edits', async () => {
  const b = browser(), request = deferred();
  b.get('transcript').value = 'original transcript';
  b.sandbox.handleFetch = () => request.promise;
  const analyzing = b.get('analyzeBtn').click();
  b.get('transcript').value = 'later edit';
  request.resolve(response({risk_score:.4,action:'warn',scam_score:.4,spoof_used:false,reasoning_log:[]}));
  await analyzing;
  assert.equal(b.run('historyEntries[0].transcript'), 'original transcript');
});

test('stopped microphone session is saved once, including unconfirmed and interim text', async () => {
  const b = browser();
  await b.run('startLiveCall()');
  b.run('recordFailure(currentCall); sendTurn(currentCall,"queued speech"); currentCall.interimText="interim speech"; globalThis.savedCall=currentCall; stopLiveCall(); releaseCall(savedCall)');
  assert.equal(b.run('historyEntries.length'), 1);
  assert.equal(b.run('historyEntries[0].transcript'), 'queued speech interim speech');
  assert.equal(b.run('historyEntries[0].unconfirmed'), true);
  assert.equal(b.run('historyEntries[0].risk'), null);
});

test('microphone startup failure does not create an empty history entry', async () => {
  const b = browser();
  b.sandbox.navigator.mediaDevices.getUserMedia = async () => { throw new Error('denied'); };
  await b.run('startLiveCall()');
  assert.equal(b.run('historyEntries.length'), 0);
});

test('history delete, bounded retention, and confirmed clear do not affect capture', async () => {
  const storage = new Map(), b = browser(storage);
  for(let i=0;i<30;i++) b.run(`saveHistory({id:'${i}',timestamp:1000,mode:'transcript',transcript:'text',risk:0,action:'monitor',unconfirmed:false})`);
  assert.equal(b.run('historyEntries.length'), 25);
  assert.equal(b.run('historyEntries[0].id'), '29');
  b.run('deleteHistory("29")');
  assert.equal(b.run('historyEntries.length'), 24);
  await b.run('startLiveCall()');
  b.sandbox.window.confirm = () => false;
  b.get('clearHistory').click();
  assert.equal(b.run('historyEntries.length'), 24);
  b.sandbox.window.confirm = () => true;
  b.get('clearHistory').click();
  assert.equal(b.run('historyEntries.length'), 0);
  assert.equal(storage.has('echoguard.history.v1'), false);
  assert.equal(b.run('liveActive'), true);
  b.run('stopLiveCall()');
});

test('malformed saved history is not overwritten until the user clears it', () => {
  const storage = new Map([['echoguard.history.v1','bad json']]), b = browser(storage);
  b.run(`saveHistory({id:'new',timestamp:1000,mode:'transcript',transcript:'text',risk:0,action:'monitor',unconfirmed:false})`);
  assert.equal(storage.get('echoguard.history.v1'), 'bad json');
  assert.equal(b.run('historyEntries.length'), 1);
  assert.match(b.get('historyStatus').textContent, /could not be loaded/);
  b.get('clearHistory').click();
  assert.equal(storage.has('echoguard.history.v1'), false);
});

test('storage quota failure leaves full text available and shows a warning', () => {
  const b = browser();
  b.sandbox.localStorage.setItem = () => { throw new Error('quota'); };
  b.run(`saveHistory({id:'new',timestamp:1000,mode:'transcript',transcript:'full text',risk:0,action:'monitor',unconfirmed:false}); openHistory('new')`);
  assert.equal(b.get('historyTranscript').textContent, 'full text');
  assert.match(b.get('historyStatus').textContent, /only available on this page/);
});

test('oversized history transcripts are rejected instead of silently truncated', () => {
  const b = browser();
  b.run(`saveHistory({id:'new',timestamp:1000,mode:'transcript',transcript:'x'.repeat(50001),risk:0,action:'monitor',unconfirmed:false})`);
  assert.equal(b.run('historyEntries.length'), 0);
  assert.match(b.get('historyStatus').textContent, /was not saved/);
});

test('browser theme preference follows toggle and survives reload', () => {
  const storage = new Map(), b = browser(storage);
  assert.equal(b.sandbox.document.documentElement.dataset.theme, 'dark');
  b.get('themeToggle').click();
  assert.equal(b.sandbox.document.documentElement.dataset.theme, 'light');
  const reloaded = browser(storage);
  assert.equal(reloaded.sandbox.document.documentElement.dataset.theme, 'light');
});

test('example handlers do not attach to theme or history controls', () => {
  assert.ok(script.includes("document.querySelectorAll('.chip[data-example]')"));
  assert.ok(!script.includes("document.querySelectorAll('.chip')"));
});

test('microphone history keeps the newest confirmed risk revision', async () => {
  const b = browser();
  await b.run('startLiveCall()');
  b.run(`currentCall.transcriptParts.push('recognized text'); applyLiveResponse(currentCall,{revision:2,risk_score:.8,action:'block'}); applyLiveResponse(currentCall,{revision:1,risk_score:.1,action:'monitor'}); stopLiveCall()`);
  assert.equal(b.run('historyEntries[0].risk'), .8);
  assert.equal(b.run('historyEntries[0].action'), 'block');
});

test('failed paste analysis is not recorded as a successful result', async () => {
  const b = browser();
  b.get('transcript').value = 'test text';
  b.sandbox.handleFetch = async () => response({error:'unavailable'},503);
  await b.get('analyzeBtn').click();
  assert.equal(b.run('historyEntries.length'), 0);
});

test('server startup failure releases microphone and allows retry', async () => {
  const b = browser();
  b.sandbox.handleFetch = async () => response({ error: 'server busy' }, 503);
  await b.run('startLiveCall()');
  assert.equal(b.streams[0].track.stopped, true);
  assert.equal(b.run('currentCall'), null);
  b.sandbox.handleFetch = null;
  await b.run('startLiveCall()');
  assert.equal(b.run('liveActive'), true);
  b.run('stopLiveCall()');
});

test('a stalled session startup times out and releases the microphone', async () => {
  const b = browser();
  b.sandbox.handleFetch = (route, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => reject(new Error('startup timed out')), { once: true });
  });
  const starting = b.run('startLiveCall()');
  await settle();
  assert.equal(b.streams[0].track.stopped, false);
  await b.advance(30001);
  await starting;
  assert.equal(b.streams[0].track.stopped, true);
  assert.equal(b.run('currentCall'), null);
  assert.match(b.get('micStatus').textContent, /startup timed out/);
});

test('stop during startup cleans the late session without touching restarted call', async () => {
  const b = browser(), late = deferred();
  let starts = 0;
  b.sandbox.handleFetch = async route => {
    if (route === '/api/live/start') return ++starts === 1 ? late.promise : response({ session_id: 'new-session' });
    return response({ ended: true });
  };
  const oldStart = b.run('startLiveCall()');
  await settle();
  b.run('stopLiveCall()');
  await b.run('startLiveCall()');
  late.resolve(response({ session_id: 'old-session' }));
  await oldStart;
  assert.equal(b.run('currentCall.sid'), 'new-session');
  assert.equal(b.streams[0].track.stopped, true);
  assert.equal(b.streams[1].track.stopped, false);
  assert(b.requests.some(r => r.route === '/api/live/end' && r.options.body.get('session_id') === 'old-session'));
  assert.equal(b.get('micStatus').textContent, 'listening…');
  b.run('stopLiveCall()');
});

test('cooldown retains transcript turns and sends them in order', async () => {
  const b = browser();
  await b.run('startLiveCall()');
  b.run(`recordFailure(currentCall); sendTurn(currentCall,'first'); sendTurn(currentCall,'second')`);
  assert.equal(b.run('currentCall.pending.length'), 2);
  assert.equal(b.requests.filter(r => r.route === '/api/live/turn').length, 0);
  await b.advance(1001);
  assert.equal(b.run('currentCall.pending.length'), 0);
  assert.deepEqual(b.requests.filter(r => r.route === '/api/live/turn').map(r => r.options.body.get('text')), ['first', 'second']);
  b.run('stopLiveCall()');
});

test('lost response is retried with the same turn ID, not duplicate text', async () => {
  const b = browser();
  await b.run('startLiveCall()');
  let attempts = 0;
  b.sandbox.handleFetch = async route => {
    if (route === '/api/live/turn' && ++attempts === 1) throw new Error('lost response');
    return response({ risk_score: 0.4, scam_score: 0.4, action: 'warn', revision: 1 });
  };
  b.run(`sendTurn(currentCall,'first')`);
  await settle();
  assert.equal(b.run('currentCall.pending.length'), 1);
  await b.advance(1001);
  const turns = b.requests.filter(r => r.route === '/api/live/turn');
  assert.equal(turns.length, 2);
  assert.equal(turns[0].options.body.get('turn_id'), turns[1].options.body.get('turn_id'));
  assert.equal(b.run('currentCall.pending.length'), 0);
  assert.equal(b.get('liveTranscript').children.length, 1);
  b.run('stopLiveCall()');
});

test('late audio response after restart cannot change the new call', async () => {
  const b = browser(), late = deferred();
  await b.run('startLiveCall()');
  b.sandbox.handleFetch = async route => route === '/api/live/audio-chunk' ? late.promise : response({ session_id: 'new-session' });
  const request = b.run('sendAudioChunk(currentCall,new Float32Array(512),16000)');
  await settle();
  b.run('stopLiveCall()');
  await b.run('startLiveCall()');
  late.resolve(response({ risk_score: 0.99, action: 'block', revision: 9 }));
  await request;
  assert.equal(b.get('riskReadoutLive').textContent, '0%');
  assert.equal(b.run('currentCall.sid'), 'new-session');
  assert.equal(b.run('currentCall.audioBusy'), false);
  assert.equal(b.contexts[0].closed, true);
  b.run('stopLiveCall()');
});

test('out-of-order responses cannot lower the latest risk display', async () => {
  const b = browser();
  await b.run('startLiveCall()');
  b.run(`applyLiveResponse(currentCall,{revision:2,risk_score:0.9,action:'block'});
         applyLiveResponse(currentCall,{revision:1,risk_score:0.1,action:'monitor'});`);
  assert.equal(b.get('riskReadoutLive').textContent, '90%');
  assert.equal(b.get('verdictBadgeLive').textContent, 'block');
  b.run('stopLiveCall()');
});

test('expired session stops capture instead of silently dropping transcript', async () => {
  const b = browser();
  await b.run('startLiveCall()');
  b.sandbox.handleFetch = async () => response({ error: 'session expired' }, 404);
  b.run(`sendTurn(currentCall,'visible but unsent')`);
  await settle();
  assert.equal(b.run('currentCall'), null);
  assert.equal(b.streams[0].track.stopped, true);
  assert.match(b.get('micStatus').textContent, /session expired/);
  assert.equal(b.get('liveTranscript').children[0].textContent, 'visible but unsent');
});

test('a hung turn times out and remains queued for retry', async () => {
  const b = browser();
  await b.run('startLiveCall()');
  b.sandbox.handleFetch = (route, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => reject(new Error('timeout')), { once: true });
  });
  b.run(`sendTurn(currentCall,'still queued')`);
  await settle();
  assert.equal(b.run('currentCall.turnBusy'), true);
  await b.advance(30001);
  assert.equal(b.run('currentCall.turnBusy'), false);
  assert.equal(b.run('currentCall.pending.length'), 1);
  b.sandbox.handleFetch = null;
  b.run('stopLiveCall()');
});

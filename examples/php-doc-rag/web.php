<?php
/**
 * PHP Documentation RAG — Web Chat Interface
 *
 * Multi-turn conversational search over the PHP manual. Streams answers
 * token-by-token via SSE. Conversation history is kept client-side and
 * sent with each request so the model can reference prior exchanges.
 *
 * Access at: http://php.localhost/php-doc-rag/web.php
 */

$dir = __DIR__;
$indexPath = "$dir/data/index.bin";
$embedPath = "$dir/models/nomic-embed-text-v1.5-q8_0.gguf";
$genPath   = "$dir/models/qwen2.5-3b-instruct-q4_k_m.gguf";
$loraPath  = "$dir/models/qwen-rag-lora.gguf";

// --- Handle chat API (POST with JSON body) ---
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: text/event-stream');
    header('Cache-Control: no-cache');
    header('X-Accel-Buffering: no');

    ignore_user_abort(false);
    ini_set('log_errors', '1');
    ini_set('error_log', __DIR__ . '/data/error.log');

    $input = json_decode(file_get_contents('php://input'), true);
    $query = trim($input['query'] ?? '');
    $history = $input['history'] ?? []; // [{role, content}, ...]

    if ($query === '') {
        sendSSE('error', 'Empty query');
        exit;
    }

    $chunks = loadIndex($indexPath);

    // Load models (persistent across requests via ext-llama caching)
    // GPU layers: set to -1 (all) for GPU, 0 for CPU-only.
    // Each php-fpm worker gets its own VRAM copy, so with 4 workers
    // and a 2.4GB model that's ~10GB. Adjust based on available VRAM.
    $gpuLayers = (int)(getenv('LLAMA_GPU_LAYERS') ?: 0);

    $embedModel = new Llama\Model($embedPath, ['n_gpu_layers' => $gpuLayers]);
    $embedCtx = new Llama\Context($embedModel, [
        'n_ctx' => 512, 'n_batch' => 512, 'embeddings' => true, 'n_threads' => 4,
    ]);

    $genModel = new Llama\Model($genPath, ['n_gpu_layers' => $gpuLayers]);
    $genCtx = new Llama\Context($genModel, ['n_ctx' => 4096, 'n_threads' => 4]);

    if (file_exists($loraPath)) {
        $lora = new Llama\LoRA($genModel, $loraPath);
        $genCtx->applyLoRA($lora);
    }

    // RAG retrieval on the current query
    $queryEmb = $embedCtx->embed("search_query: $query");
    $scores = [];
    foreach ($chunks as $i => $chunk) {
        $scores[$i] = cosineSim($queryEmb, $chunk['embedding']);
    }
    arsort($scores);
    $topResults = [];
    foreach (array_slice($scores, 0, 5, true) as $i => $score) {
        $topResults[] = ['title' => $chunks[$i]['title'], 'text' => $chunks[$i]['text'], 'score' => $score];
    }

    sendSSE('sources', json_encode(array_map(fn($r) => [
        'title' => $r['title'],
        'score' => round($r['score'], 4),
    ], $topResults)));

    // Build multi-turn prompt with ChatML
    $context = '';
    foreach ($topResults as $r) {
        $context .= "## {$r['title']}\n{$r['text']}\n\n";
    }

    // Build prompt, fitting within context window.
    // Use the actual tokenizer to count tokens, not char heuristics.
    $maxTokens = 4096 - 512; // reserve 512 for the response

    $systemText = "You are a PHP documentation assistant. Answer questions based on the " .
        "provided documentation excerpts. Be concise and include code examples " .
        "when relevant. If the documentation doesn't cover the question, say so.";

    $system = "<|im_start|>system\n$systemText\n\nDocumentation:\n$context<|im_end|>\n";
    $tail = "<|im_start|>user\n$query<|im_end|>\n<|im_start|>assistant\n";

    $fixedTokens = count($genModel->tokenize($system . $tail, false, true));

    // Build history block, checking token count
    $historyBlock = '';
    if (count($history) > 0) {
        // Try full history first
        $candidate = '';
        foreach ($history as $msg) {
            $role = $msg['role'] === 'assistant' ? 'assistant' : 'user';
            $candidate .= "<|im_start|>$role\n{$msg['content']}<|im_end|>\n";
        }

        if ($fixedTokens + count($genModel->tokenize($candidate, false, true)) <= $maxTokens) {
            // Full history fits
            $historyBlock = $candidate;
        } elseif (count($history) > 2) {
            // Summarize older turns, keep last exchange verbatim
            sendSSE('status', 'Summarizing conversation...');

            $older = array_slice($history, 0, -2);
            $recent = array_slice($history, -2);
            $summary = summarizeHistory($older, $genCtx);

            $historyBlock = "<|im_start|>user\n[Prior discussion: $summary]<|im_end|>\n";
            foreach ($recent as $msg) {
                $role = $msg['role'] === 'assistant' ? 'assistant' : 'user';
                $historyBlock .= "<|im_start|>$role\n{$msg['content']}<|im_end|>\n";
            }

            // Verify it fits now; if not, drop the summary and use only last exchange
            if ($fixedTokens + count($genModel->tokenize($historyBlock, false, true)) > $maxTokens) {
                $historyBlock = '';
                foreach ($recent as $msg) {
                    $role = $msg['role'] === 'assistant' ? 'assistant' : 'user';
                    $historyBlock .= "<|im_start|>$role\n{$msg['content']}<|im_end|>\n";
                }
            }
        }

        // Final safety: if still too long, drop history entirely
        if ($fixedTokens + count($genModel->tokenize($historyBlock, false, true)) > $maxTokens) {
            $historyBlock = '';
        }
    }

    $formatted = $system . $historyBlock . $tail;

    // Stream answer
    foreach ($genCtx->stream($formatted, [
        'max_tokens' => 512,
        'temperature' => 0.2,
        'parse_special' => true,
    ]) as $piece) {
        sendSSE('token', $piece);
    }

    sendSSE('done', '');
    exit;
}

// --- Helper functions ---
function loadIndex(string $path): array {
    $fp = fopen($path, 'rb');
    ['count' => $count, 'dims' => $dims] = unpack('Vcount/Vdims', fread($fp, 8));
    $chunks = [];
    for ($i = 0; $i < $count; $i++) {
        $titleLen = unpack('V', fread($fp, 4))[1];
        $title = fread($fp, $titleLen);
        $textLen = unpack('V', fread($fp, 4))[1];
        $text = fread($fp, $textLen);
        $embedding = array_values(unpack('f*', fread($fp, $dims * 4)));
        $chunks[] = compact('title', 'text', 'embedding');
    }
    fclose($fp);
    return $chunks;
}

function cosineSim(array $a, array $b): float {
    $dot = $na = $nb = 0.0;
    for ($i = 0, $n = count($a); $i < $n; $i++) {
        $dot += $a[$i] * $b[$i]; $na += $a[$i] * $a[$i]; $nb += $b[$i] * $b[$i];
    }
    return $dot / (sqrt($na) * sqrt($nb));
}

function summarizeHistory(array $turns, Llama\Context $ctx): string {
    $conversation = '';
    foreach ($turns as $msg) {
        $role = $msg['role'] === 'user' ? 'User' : 'Assistant';
        // Truncate very long individual messages for the summary input
        $content = $msg['content'];
        if (strlen($content) > 600) {
            $content = substr($content, 0, 600) . '...';
        }
        $conversation .= "$role: $content\n\n";
    }

    return $ctx->chat([
        ['role' => 'system', 'content' => 'Summarize this conversation in 2-3 sentences. Focus on what PHP topics were discussed and what was learned. Be very concise.'],
        ['role' => 'user', 'content' => $conversation],
    ], ['max_tokens' => 150, 'temperature' => 0.1]);
}

function sendSSE(string $event, string $data): void {
    echo "event: $event\ndata: " . json_encode($data) . "\n\n";
    ob_flush();
    flush();
}

// --- Serve the HTML UI ---
?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PHP Doc Chat</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/github-dark-dimmed.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/languages/php.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e; color: #e0e0e0;
    height: 100vh; display: flex; flex-direction: column;
  }

  /* Header */
  .header {
    padding: 1rem; text-align: center; border-bottom: 1px solid #222;
    flex-shrink: 0;
  }
  .header h1 { font-size: 1.25rem; color: #7b68ee; display: inline; }
  .header span { color: #555; font-size: 0.85rem; margin-left: 0.5rem; }
  .header .new-chat {
    float: right; background: none; border: 1px solid #333; color: #888;
    padding: 0.3rem 0.8rem; border-radius: 6px; cursor: pointer; font-size: 0.8rem;
  }
  .header .new-chat:hover { border-color: #7b68ee; color: #7b68ee; }

  /* Chat area */
  .chat {
    flex: 1; overflow-y: auto; padding: 1rem; max-width: 800px;
    width: 100%; margin: 0 auto;
  }
  .chat::-webkit-scrollbar { width: 6px; }
  .chat::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

  .message { margin-bottom: 1.5rem; }
  .message.user .bubble {
    background: #7b68ee; color: white; border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1rem; display: inline-block; max-width: 80%; float: right;
  }
  .message.user::after { content: ''; display: block; clear: both; }
  .message.assistant .sources {
    display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.5rem;
  }
  .source-tag {
    font-size: 0.7rem; padding: 0.2rem 0.5rem; border-radius: 4px;
    background: #16213e; border: 1px solid #2a2a4a; color: #777;
  }
  .source-tag .score { color: #7b68ee; margin-left: 0.25rem; }
  .message.assistant .bubble {
    background: #16213e; border: 1px solid #2a2a4a; border-radius: 12px 12px 12px 4px;
    padding: 1rem 1.25rem; line-height: 1.7; white-space: pre-wrap; font-size: 0.93rem;
  }
  .message .meta {
    font-size: 0.7rem; color: #444; margin-top: 0.3rem;
  }
  .message.user .meta { text-align: right; }

  /* Code blocks */
  .bubble code {
    background: #0f3460; padding: 0.15rem 0.4rem; border-radius: 4px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.85em;
  }
  .bubble pre {
    background: #0f3460; padding: 1rem; border-radius: 8px; overflow-x: auto;
    margin: 0.75rem 0;
  }
  .bubble pre code { background: none; padding: 0; }
  .bubble pre code.hljs { background: none; padding: 0; }

  .cursor { animation: blink 1s infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  /* Empty state */
  .empty {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; height: 100%; color: #444; text-align: center;
  }
  .empty h2 { color: #7b68ee; margin-bottom: 0.5rem; font-size: 1.5rem; }
  .empty p { max-width: 400px; line-height: 1.6; }
  .examples { margin-top: 1.5rem; display: flex; flex-direction: column; gap: 0.5rem; }
  .examples button {
    background: #16213e; border: 1px solid #2a2a4a; color: #999; padding: 0.6rem 1rem;
    border-radius: 8px; cursor: pointer; font-size: 0.85rem; text-align: left;
  }
  .examples button:hover { border-color: #7b68ee; color: #ccc; }

  /* Input bar */
  .input-bar {
    flex-shrink: 0; border-top: 1px solid #222; padding: 0.75rem 1rem;
    max-width: 800px; width: 100%; margin: 0 auto;
  }
  .input-bar form { display: flex; gap: 0.5rem; }
  .input-bar input {
    flex: 1; padding: 0.75rem 1rem; border: 1px solid #333; border-radius: 8px;
    background: #16213e; color: #e0e0e0; font-size: 1rem; outline: none;
  }
  .input-bar input:focus { border-color: #7b68ee; }
  .input-bar button {
    padding: 0.75rem 1.25rem; border: none; border-radius: 8px;
    background: #7b68ee; color: white; font-size: 1rem; cursor: pointer;
  }
  .input-bar button:hover { background: #6c5ce7; }
  .input-bar button:disabled { background: #444; cursor: wait; }

  .powered {
    text-align: center; font-size: 0.7rem; color: #333; padding: 0.4rem;
  }
</style>
</head>
<body>

<div class="header">
  <h1>PHP Doc Chat</h1>
  <span>local AI answers from the official PHP docs</span>
  <button class="new-chat" onclick="newChat()">New Chat</button>
</div>

<div class="chat" id="chat">
  <div class="empty" id="empty">
    <h2>Ask anything about PHP</h2>
    <p>Answers are generated from the official PHP documentation using local models running on this machine.</p>
    <div class="examples">
      <button onclick="askExample(this)">How do I sort an array by value?</button>
      <button onclick="askExample(this)">What's the difference between == and ===?</button>
      <button onclick="askExample(this)">How do I read a file line by line?</button>
      <button onclick="askExample(this)">Explain PHP's match expression</button>
    </div>
  </div>
</div>

<div class="input-bar">
  <form onsubmit="send(event)">
    <input type="text" id="q" placeholder="Ask about PHP..." autofocus>
    <button type="submit" id="btn">Send</button>
  </form>
</div>
<div class="powered">Qwen2.5-3B + RAG LoRA &bull; nomic-embed-text &bull; ext-llama</div>

<script>
const chatEl = document.getElementById('chat');
const emptyEl = document.getElementById('empty');
const qEl = document.getElementById('q');
const btnEl = document.getElementById('btn');

let history = []; // [{role: 'user'|'assistant', content: '...'}]
let generating = false;
let abortController = null;

function askExample(el) {
  qEl.value = el.textContent;
  send(new Event('submit'));
}

function newChat() {
  history = [];
  chatEl.innerHTML = '';
  chatEl.appendChild(emptyEl);
  emptyEl.style.display = 'flex';
  qEl.value = '';
  qEl.focus();
}

function send(e) {
  e.preventDefault();
  const q = qEl.value.trim();
  if (!q || generating) return;

  emptyEl.style.display = 'none';
  // Abort any in-flight request
  if (abortController) abortController.abort();
  abortController = new AbortController();

  generating = true;
  btnEl.disabled = true;
  btnEl.textContent = '...';
  qEl.value = '';

  // Add user message
  const userMsg = addMessage('user', q);

  // Add assistant placeholder
  const assistantMsg = addMessage('assistant', '');
  const bubble = assistantMsg.querySelector('.bubble');
  const sourcesEl = assistantMsg.querySelector('.sources');
  const metaEl = assistantMsg.querySelector('.meta');
  bubble.innerHTML = '<span class="cursor">|</span>';

  const t0 = performance.now();
  let text = '';

  // POST with conversation history
  fetch('web.php', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ query: q, history: history }),
    signal: abortController.signal,
  }).then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    function read() {
      reader.read().then(({done, value}) => {
        if (done) { finish(); return; }
        buffer += decoder.decode(value, {stream: true});

        // Parse SSE events from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete line

        let eventType = null;
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7);
          } else if (line.startsWith('data: ') && eventType) {
            const data = JSON.parse(line.slice(6));
            handleEvent(eventType, data);
            eventType = null;
          }
        }

        read();
      });
    }

    let tokenCount = 0;
    function handleEvent(event, data) {
      if (event === 'sources') {
        const sources = JSON.parse(data);
        sourcesEl.innerHTML = sources.map(s =>
          `<span class="source-tag">${esc(s.title)}<span class="score">${s.score}</span></span>`
        ).join('');
      } else if (event === 'status') {
        bubble.innerHTML = `<span style="color:#666">${esc(data)}</span><span class="cursor">|</span>`;
      } else if (event === 'token') {
        text += data;
        tokenCount++;
        bubble.innerHTML = formatMarkdown(text) + '<span class="cursor">|</span>';
        // Highlight completed code blocks periodically
        if (tokenCount % 20 === 0) highlightAll(bubble);
        chatEl.scrollTop = chatEl.scrollHeight;
      } else if (event === 'done') {
        finish();
      }
    }

    function finish() {
      bubble.innerHTML = formatMarkdown(text);
      highlightAll(bubble);
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      metaEl.textContent = secs + 's';

      // Save to history
      history.push({role: 'user', content: q});
      history.push({role: 'assistant', content: text});

      generating = false;
      abortController = null;
      btnEl.disabled = false;
      btnEl.textContent = 'Send';
      qEl.focus();
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    read();
  }).catch(err => {
    if (err.name === 'AbortError') return; // intentional abort for new message
    bubble.innerHTML = 'Error connecting to server.';
    generating = false;
    abortController = null;
    btnEl.disabled = false;
    btnEl.textContent = 'Send';
  });

  chatEl.scrollTop = chatEl.scrollHeight;
}

function addMessage(role, content) {
  const div = document.createElement('div');
  div.className = 'message ' + role;
  if (role === 'user') {
    div.innerHTML = `<div class="bubble">${esc(content)}</div><div class="meta"></div>`;
  } else {
    div.innerHTML =
      `<div class="sources"></div>` +
      `<div class="bubble"></div>` +
      `<div class="meta"></div>`;
  }
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function formatMarkdown(text) {
  return esc(text)
    .replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
      const cls = lang ? `language-${lang}` : 'language-php';
      return `<pre><code class="${cls}">${code}</code></pre>`;
    })
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

function highlightAll(el) {
  el.querySelectorAll('pre code').forEach(block => {
    block.removeAttribute('data-highlighted');
    hljs.highlightElement(block);
  });
}
</script>
</body>
</html>

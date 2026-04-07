<?php
/**
 * PHP Documentation RAG Search
 *
 * Embeds a user query, finds the most relevant PHP manual pages via
 * cosine similarity, then uses Qwen2.5-3B + RAG LoRA to generate
 * an answer grounded in the documentation.
 *
 * Usage: php search.php "How do I sort an array by value?"
 *        php search.php --interactive
 */

$dir = __DIR__;
$indexPath    = "$dir/data/index.bin";
$embedPath    = "$dir/models/nomic-embed-text-v1.5-q8_0.gguf";
$genPath      = "$dir/models/qwen2.5-3b-instruct-q4_k_m.gguf";
$loraPath     = "$dir/models/qwen-rag-lora.gguf";

if (!file_exists($indexPath)) {
    die("Index not found. Run: php index.php\n");
}

// --- Load index ---
$fp = fopen($indexPath, 'rb');
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

fprintf(STDERR, "Loaded %d indexed pages (%d dimensions).\n", $count, $dims);

// --- Load models ---
fprintf(STDERR, "Loading models...\n");
$embedModel = new Llama\Model($embedPath);
$embedCtx = new Llama\Context($embedModel, [
    'n_ctx' => 512,
    'embeddings' => true,
    'n_threads' => 4,
]);

$genModel = new Llama\Model($genPath);
$genCtx = new Llama\Context($genModel, ['n_ctx' => 4096, 'n_threads' => 4]);

if (file_exists($loraPath)) {
    $lora = new Llama\LoRA($genModel, $loraPath);
    $genCtx->applyLoRA($lora);
    fprintf(STDERR, "RAG LoRA adapter applied.\n");
}

fprintf(STDERR, "Ready.\n\n");

// --- Search function ---
function search(string $query, array $chunks, Llama\Context $embedCtx, int $topK = 5): array
{
    // Embed query with search_query prefix (nomic convention)
    $queryEmb = $embedCtx->embed("search_query: $query");

    // Compute cosine similarity against all chunks
    $scores = [];
    foreach ($chunks as $i => $chunk) {
        $scores[$i] = cosineSim($queryEmb, $chunk['embedding']);
    }

    arsort($scores);

    $results = [];
    foreach (array_slice($scores, 0, $topK, true) as $i => $score) {
        $results[] = [
            'title' => $chunks[$i]['title'],
            'text'  => $chunks[$i]['text'],
            'score' => $score,
        ];
    }

    return $results;
}

function cosineSim(array $a, array $b): float
{
    $dot = $na = $nb = 0.0;
    for ($i = 0, $n = count($a); $i < $n; $i++) {
        $dot += $a[$i] * $b[$i];
        $na  += $a[$i] * $a[$i];
        $nb  += $b[$i] * $b[$i];
    }
    return $dot / (sqrt($na) * sqrt($nb));
}

function answer(string $query, array $results, Llama\Context $ctx): string
{
    // Build context from top results
    $context = '';
    foreach ($results as $r) {
        $context .= "## {$r['title']}\n{$r['text']}\n\n";
    }

    $messages = [
        ['role' => 'system', 'content' =>
            "You are a PHP documentation assistant. Answer the user's question " .
            "based ONLY on the provided documentation excerpts. Be concise and " .
            "include code examples when relevant. If the documentation doesn't " .
            "cover the question, say so."
        ],
        ['role' => 'user', 'content' =>
            "Documentation:\n\n$context\n---\nQuestion: $query"
        ],
    ];

    return $ctx->chat($messages, [
        'max_tokens' => 512,
        'temperature' => 0.2,
    ]);
}

// --- Main ---
$interactive = in_array('--interactive', $argv);
$query = null;

if (!$interactive) {
    // Single query from command line
    $args = array_filter($argv, fn($a) => $a !== '--interactive' && $a !== $argv[0]);
    $query = implode(' ', $args);
    if (empty(trim($query))) {
        echo "Usage: php search.php \"your question about PHP\"\n";
        echo "       php search.php --interactive\n";
        exit(1);
    }
}

do {
    if ($interactive) {
        echo "\033[1mQuestion:\033[0m ";
        $query = trim(fgets(STDIN));
        if ($query === '' || $query === 'quit' || $query === 'exit') break;
    }

    // Search
    $t0 = microtime(true);
    $results = search($query, $chunks, $embedCtx);
    $searchTime = microtime(true) - $t0;

    // Show retrieved docs
    echo "\n\033[2m";
    foreach ($results as $i => $r) {
        printf("  [%d] %.4f  %s\n", $i + 1, $r['score'], $r['title']);
    }
    printf("  Search: %.0fms\033[0m\n\n", $searchTime * 1000);

    // Generate answer
    echo "\033[1mAnswer:\033[0m ";
    $t0 = microtime(true);

    // Stream the answer token by token
    $context = '';
    foreach ($results as $r) {
        $context .= "## {$r['title']}\n{$r['text']}\n\n";
    }
    $messages = [
        ['role' => 'system', 'content' =>
            "You are a PHP documentation assistant. Answer the user's question " .
            "based ONLY on the provided documentation excerpts. Be concise and " .
            "include code examples when relevant. If the documentation doesn't " .
            "cover the question, say so."
        ],
        ['role' => 'user', 'content' => "Documentation:\n\n$context\n---\nQuestion: $query"],
    ];

    // Apply chat template manually for streaming
    // (stream() takes a raw prompt, not messages)
    $tmpl = $genModel->chatTemplate();
    $formatted = applyTemplate($tmpl, $messages);

    foreach ($genCtx->stream($formatted, ['max_tokens' => 512, 'temperature' => 0.2, 'parse_special' => true]) as $piece) {
        echo $piece;
    }

    $genTime = microtime(true) - $t0;
    printf("\n\033[2m  Generation: %.1fs\033[0m\n\n", $genTime);

} while ($interactive);

/**
 * Minimal chat template formatter.
 * Works for Qwen2.5's ChatML-style template.
 */
function applyTemplate(string $tmpl, array $messages): string
{
    // Qwen uses ChatML: <|im_start|>role\ncontent<|im_end|>
    $out = '';
    foreach ($messages as $msg) {
        $out .= "<|im_start|>{$msg['role']}\n{$msg['content']}<|im_end|>\n";
    }
    $out .= "<|im_start|>assistant\n";
    return $out;
}

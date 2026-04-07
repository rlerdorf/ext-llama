<?php
/**
 * Index the PHP documentation for RAG search.
 *
 * Reads each HTML file from the PHP manual, strips tags to extract plain text,
 * generates an embedding vector using nomic-embed-text, and stores the index
 * as a binary file for fast loading.
 *
 * Usage: php index.php [--limit N]
 *
 * The index is stored in data/index.bin as a packed binary format:
 *   [uint32 count][uint32 dims]
 *   For each chunk: [uint32 title_len][title][uint32 text_len][text][float32 * dims]
 */

$dir = __DIR__;
$manualDir = "$dir/data/php-manual";
$modelPath = "$dir/models/nomic-embed-text-v1.5-q8_0.gguf";
$indexPath = "$dir/data/index.bin";

if (!is_dir($manualDir)) {
    die("PHP manual not found. Run: bash download.sh\n");
}
if (!file_exists($modelPath)) {
    die("Embedding model not found. Run: bash download.sh\n");
}

$limit = null;
if (($idx = array_search('--limit', $argv)) !== false && isset($argv[$idx + 1])) {
    $limit = (int)$argv[$idx + 1];
}

// Patterns for pages worth indexing (functions, classes, core language)
$patterns = [
    'function.*.html',
    'class.*.html',
    'language.*.html',
    'control-structures.*.html',
    'ref.*.html',
    'book.*.html',
    'intro.*.html',
];

// Collect matching files
$files = [];
foreach ($patterns as $pattern) {
    $files = array_merge($files, glob("$manualDir/$pattern"));
}
$files = array_unique($files);
sort($files);

if ($limit) {
    $files = array_slice($files, 0, $limit);
}

printf("Found %d documentation pages to index.\n", count($files));

/**
 * Extract clean text from a PHP manual HTML page.
 * Returns [title, body_text] or null if the page has no useful content.
 */
function extractText(string $html): ?array
{
    // Remove script/style blocks
    $html = preg_replace('#<(script|style)[^>]*>.*?</\1>#si', '', $html);

    // Extract title
    $title = '';
    if (preg_match('#<title>([^<]+)</title>#i', $html, $m)) {
        $title = html_entity_decode(trim($m[1]), ENT_QUOTES | ENT_HTML5, 'UTF-8');
        // Strip " - Manual" suffix
        $title = preg_replace('# - Manual$#', '', $title);
    }

    // Remove navigation, header, footer
    $html = preg_replace('#<nav[^>]*>.*?</nav>#si', '', $html);
    $html = preg_replace('#<div\s+class="navbar[^"]*"[^>]*>.*?</div>#si', '', $html);

    // Strip remaining tags, decode entities
    $text = strip_tags($html);
    $text = html_entity_decode($text, ENT_QUOTES | ENT_HTML5, 'UTF-8');

    // Normalize whitespace
    $text = preg_replace('/[ \t]+/', ' ', $text);
    $text = preg_replace("/\n{3,}/", "\n\n", $text);
    $text = trim($text);

    // Skip near-empty pages
    if (strlen($text) < 100) {
        return null;
    }

    // Truncate very long pages to ~2000 chars (embedding model context limit)
    if (strlen($text) > 2000) {
        $text = substr($text, 0, 2000);
        // Don't cut mid-word
        $text = preg_replace('/\s+\S*$/', '', $text);
    }

    return [$title, $text];
}

// Load embedding model
printf("Loading embedding model...\n");
$embedModel = new Llama\Model($modelPath);
$embedCtx = new Llama\Context($embedModel, [
    'n_ctx' => 8192,
    'n_batch' => 8192,
    'embeddings' => true,
    'n_threads' => 4,
]);
$dims = $embedModel->nEmbd();
printf("Embedding dimensions: %d\n", $dims);

// Process files and build index
$chunks = [];
$t0 = microtime(true);
$processed = 0;
$skipped = 0;

foreach ($files as $i => $file) {
    $html = file_get_contents($file);
    $extracted = extractText($html);

    if (!$extracted) {
        $skipped++;
        continue;
    }

    [$title, $text] = $extracted;

    // Embed the text with a search_document prefix (nomic convention)
    $embedding = $embedCtx->embed("search_document: $title\n$text");

    $chunks[] = [
        'title' => $title,
        'text' => $text,
        'embedding' => $embedding,
    ];

    $processed++;
    if ($processed % 100 === 0) {
        $elapsed = microtime(true) - $t0;
        $rate = $processed / $elapsed;
        $eta = (count($files) - $i) / $rate;
        printf("\r  Indexed %d/%d pages (%.1f pages/sec, ETA %.0fs)...",
            $processed, count($files), $rate, $eta);
    }
}

$elapsed = microtime(true) - $t0;
printf("\r  Indexed %d pages in %.1fs (%.1f pages/sec). Skipped %d empty pages.\n",
    $processed, $elapsed, $processed / $elapsed, $skipped);

// Write binary index
printf("Writing index to %s...\n", basename($indexPath));

$fp = fopen($indexPath, 'wb');
fwrite($fp, pack('VV', count($chunks), $dims));

foreach ($chunks as $chunk) {
    $title = $chunk['title'];
    $text = $chunk['text'];
    fwrite($fp, pack('V', strlen($title)));
    fwrite($fp, $title);
    fwrite($fp, pack('V', strlen($text)));
    fwrite($fp, $text);
    // Pack embedding as float32 array
    fwrite($fp, pack('f*', ...$chunk['embedding']));
}

fclose($fp);

$size = filesize($indexPath);
printf("Done! Index: %d chunks, %s\n", count($chunks), formatBytes($size));

function formatBytes(int $bytes): string
{
    if ($bytes >= 1 << 20) return sprintf('%.1f MB', $bytes / (1 << 20));
    if ($bytes >= 1 << 10) return sprintf('%.1f KB', $bytes / (1 << 10));
    return "$bytes bytes";
}

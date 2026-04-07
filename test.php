<?php
/**
 * Basic smoke test for the llama PHP extension.
 *
 * Usage: php -dextension=modules/llama.so test.php /path/to/model.gguf
 */

$model_path = $argv[1] ?? '/tmp/tinyllama.gguf';

if (!file_exists($model_path)) {
    die("Model not found: {$model_path}\n");
}

echo "Loading model: {$model_path}\n";
$model = new Llama\Model($model_path);

echo "Description: {$model->desc()}\n";
echo "Size: " . number_format($model->size()) . " bytes\n";
echo "Parameters: " . number_format($model->nParams()) . "\n";
echo "Embedding dim: {$model->nEmbd()}\n";
echo "Layers: {$model->nLayer()}\n";
echo "Chat template: " . ($model->chatTemplate() ? 'yes' : 'no') . "\n";

// Tokenize / detokenize round-trip
$tokens = $model->tokenize("Hello, world!");
$text = $model->detokenize($tokens);
assert(trim($text) === "Hello, world!", "Round-trip failed: got '{$text}'");
echo "Tokenize round-trip: OK (" . count($tokens) . " tokens)\n";

// Text completion
$ctx = new Llama\Context($model, ['n_ctx' => 512, 'n_threads' => 4]);
$t0 = microtime(true);
$result = $ctx->complete("PHP is", ['max_tokens' => 32, 'temperature' => 0.1]);
$elapsed = microtime(true) - $t0;
echo "\nComplete: PHP is{$result}\n";
echo "Time: " . round($elapsed, 2) . "s\n";

// Chat
if ($model->chatTemplate()) {
    $t0 = microtime(true);
    $result = $ctx->chat([
        ['role' => 'user', 'content' => 'Say hello in French'],
    ], ['max_tokens' => 32, 'temperature' => 0.1]);
    $elapsed = microtime(true) - $t0;
    echo "\nChat: {$result}\n";
    echo "Time: " . round($elapsed, 2) . "s\n";
}

// Embeddings
$ctx2 = new Llama\Context($model, ['n_ctx' => 512, 'embeddings' => true]);
$embd = $ctx2->embed("Test embedding");
echo "\nEmbedding dimensions: " . count($embd) . "\n";

// Error handling
try {
    new Llama\Model("/nonexistent.gguf");
    echo "ERROR: Should have thrown\n";
} catch (Llama\Exception $e) {
    echo "\nException handling: OK ({$e->getMessage()})\n";
}

echo "\nAll tests passed!\n";

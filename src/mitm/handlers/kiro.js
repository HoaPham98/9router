const { err } = require("../logger");
const { fetchRouter, pipeTransformedEventStream } = require("./base");
const path = require("path");

// Resolve paths to translator modules (ESM) from CommonJS context
const KIRO_TO_OPENAI_PATH = path.resolve(
  __dirname, "../../../open-sse/translator/request/kiro-to-openai.js"
);
const OPENAI_TO_KIRO_PATH = path.resolve(
  __dirname, "../../../open-sse/translator/response/openai-to-kiro.js"
);

/**
 * Intercept Kiro request:
 * 1. Parse Kiro body
 * 2. Convert Kiro -> OpenAI via request translator
 * 3. Send OpenAI body to 9Router
 * 4. Pipe SSE response through OpenAI -> Kiro EventStream encoder
 */
async function intercept(req, res, bodyBuffer, mappedModel) {
  try {
    const body = JSON.parse(bodyBuffer.toString());

    // Dynamic import ESM translator modules from CommonJS
    const [{ buildOpenAIPayload }, { convertOpenAIToKiro, initKiroState }] =
      await Promise.all([
        import(KIRO_TO_OPENAI_PATH),
        import(OPENAI_TO_KIRO_PATH)
      ]);

    // Convert Kiro -> OpenAI
    const openaiBody = buildOpenAIPayload(mappedModel, body, true, null);

    // Send to 9Router
    const routerRes = await fetchRouter(openaiBody, "/v1/chat/completions", req.headers);

    // Pipe SSE with EventStream binary transformation
    const state = initKiroState();
    await pipeTransformedEventStream(routerRes, res, convertOpenAIToKiro, state);
  } catch (error) {
    err(`[Kiro] ${error.message}`);
    if (!res.headersSent) {
      res.writeHead(500, { "Content-Type": "application/json" });
    }
    res.end(JSON.stringify({ error: { message: error.message, type: "mitm_error" } }));
  }
}

module.exports = { intercept };

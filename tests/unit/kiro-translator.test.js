import { describe, it, expect } from "vitest";
import { EventStreamCodec } from "@smithy/eventstream-codec";
import { toUtf8, fromUtf8 } from "@smithy/util-utf8";
import { buildOpenAIPayload } from "../../open-sse/translator/request/kiro-to-openai.js";
import { convertOpenAIToKiro, initKiroState } from "../../open-sse/translator/response/openai-to-kiro.js";
import { buildKiroPayload } from "../../open-sse/translator/request/openai-to-kiro.js";
import { convertKiroToOpenAI } from "../../open-sse/translator/response/kiro-to-openai.js";

// ── EventStream decode helper ──────────────────────────────

const codec = new EventStreamCodec(toUtf8, fromUtf8);

/** Decode a binary EventStream frame and return { eventType, payload } */
function decodeFrame(frame) {
  const msg = codec.decode(frame);
  const eventType = msg.headers[":event-type"]?.value || "";
  const payload = JSON.parse(toUtf8(msg.body));
  return { eventType, payload };
}

function decodeFirst(result) {
  const frame = Array.isArray(result) ? result[0] : result;
  return decodeFrame(frame);
}

function decodeAll(result) {
  if (!result) return [];
  const frames = Array.isArray(result) ? result : [result];
  return frames.map(f => decodeFrame(f));
}

// ── Test fixtures ──────────────────────────────────────────

const makeKiroBody = (overrides = {}) => ({
  conversationState: {
    chatTriggerType: "MANUAL",
    conversationId: "test-conv-id",
    currentMessage: {
      userInputMessage: {
        content: "What is the weather in Paris?",
        modelId: "claude-sonnet-4-6",
        origin: "AI_EDITOR"
      }
    },
    history: [],
    ...overrides.conversationState
  },
  inferenceConfig: {
    maxTokens: 32000,
    temperature: 0.7,
    topP: 0.9
  },
  ...overrides.topLevel
});

const makeOpenAIChunk = (overrides = {}) => ({
  id: "chatcmpl-test",
  object: "chat.completion.chunk",
  created: 1717000000,
  model: "gpt-5",
  choices: [{
    index: 0,
    delta: {},
    finish_reason: null,
    ...overrides.choice
  }],
  ...overrides.topLevel
});

// ── Request: kiro → openai ────────────────────────────────

describe("buildOpenAIPayload (kiro → openai)", () => {
  it("converts a basic text message", () => {
    const result = buildOpenAIPayload("mapped-model", makeKiroBody(), true, null);

    expect(result.model).toBe("mapped-model");
    expect(result.stream).toBe(true);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toEqual({
      role: "user",
      content: "What is the weather in Paris?"
    });
  });

  it("prefers model parameter over body modelId", () => {
    const body = makeKiroBody();
    const result = buildOpenAIPayload("mapped-gpt-5", body, true, null);
    expect(result.model).toBe("mapped-gpt-5");
  });

  it("falls back to body modelId when model parameter is falsy", () => {
    const result = buildOpenAIPayload("", makeKiroBody(), true, null);
    expect(result.model).toBe("claude-sonnet-4-6");
  });

  it("strips timestamp prefix from content", () => {
    const body = makeKiroBody();
    body.conversationState.currentMessage.userInputMessage.content =
      "[Context: Current time is 2026-04-29T10:30:00.000Z]\n\nWhat is the weather?";
    const result = buildOpenAIPayload("m", body, true, null);
    expect(result.messages[0].content).toBe("What is the weather?");
  });

  it("maps inferenceConfig to OpenAI fields", () => {
    const body = makeKiroBody();
    body.inferenceConfig = { maxTokens: 8000, temperature: 0.3, topP: 0.95 };
    const result = buildOpenAIPayload("m", body, true, null);
    expect(result.max_tokens).toBe(8000);
    expect(result.temperature).toBe(0.3);
    expect(result.top_p).toBe(0.95);
  });

  it("omits inferenceConfig fields when absent", () => {
    const body = makeKiroBody({ topLevel: { inferenceConfig: undefined } });
    const result = buildOpenAIPayload("m", body, true, null);
    expect(result.max_tokens).toBeUndefined();
    expect(result.temperature).toBeUndefined();
    expect(result.top_p).toBeUndefined();
  });

  it("converts tools from currentMessage context", () => {
    const body = makeKiroBody();
    body.conversationState.currentMessage.userInputMessage.userInputMessageContext = {
      tools: [{
        toolSpecification: {
          name: "get_weather",
          description: "Get current weather for a city",
          inputSchema: {
            json: {
              type: "object",
              properties: { city: { type: "string" } },
              required: ["city"]
            }
          }
        }
      }]
    };
    const result = buildOpenAIPayload("m", body, true, null);
    expect(result.tools).toHaveLength(1);
    expect(result.tools[0]).toEqual({
      type: "function",
      function: {
        name: "get_weather",
        description: "Get current weather for a city",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
          required: ["city"]
        }
      }
    });
  });

  it("converts tool results to separate tool-role messages", () => {
    const body = makeKiroBody();
    body.conversationState.currentMessage.userInputMessage.userInputMessageContext = {
      toolResults: [{
        toolUseId: "call_abc123",
        status: "success",
        content: [{ text: "Paris: 22°C, sunny" }]
      }]
    };
    const result = buildOpenAIPayload("m", body, true, null);
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toEqual({
      role: "tool",
      tool_call_id: "call_abc123",
      content: "Paris: 22°C, sunny"
    });
    expect(result.messages[1]).toEqual({
      role: "user",
      content: "What is the weather in Paris?"
    });
  });

  it("converts images to base64 data URI content parts", () => {
    const body = makeKiroBody();
    body.conversationState.currentMessage.userInputMessage.images = [{
      format: "png",
      source: { bytes: "aGVsbG8=" }
    }];
    const result = buildOpenAIPayload("m", body, true, null);
    expect(Array.isArray(result.messages[0].content)).toBe(true);
    expect(result.messages[0].content[0]).toEqual({
      type: "image_url",
      image_url: { url: "data:image/png;base64,aGVsbG8=" }
    });
    expect(result.messages[0].content[1]).toEqual({
      type: "text",
      text: "What is the weather in Paris?"
    });
  });

  it("handles images without accompanying text", () => {
    const body = makeKiroBody();
    body.conversationState.currentMessage.userInputMessage.content = "";
    body.conversationState.currentMessage.userInputMessage.images = [{
      format: "jpeg",
      source: { bytes: "aW1hZ2U=" }
    }];
    const result = buildOpenAIPayload("m", body, true, null);
    expect(Array.isArray(result.messages[0].content)).toBe(true);
    expect(result.messages[0].content).toHaveLength(1);
    expect(result.messages[0].content[0].type).toBe("image_url");
  });

  it("handles empty history gracefully", () => {
    const result = buildOpenAIPayload("m", makeKiroBody(), true, null);
    expect(result.messages).toHaveLength(1);
  });

  it("processes history items into messages", () => {
    const body = makeKiroBody();
    body.conversationState.history = [
      { userInputMessage: { content: "Hello, can you help me?", modelId: "claude-sonnet-4-6" } },
      { assistantResponseMessage: { content: "Of course! What do you need?" } },
      { userInputMessage: { content: "Tell me about Paris", modelId: "claude-sonnet-4-6" } },
      { assistantResponseMessage: { content: "Paris is the capital of France." } }
    ];
    body.conversationState.currentMessage.userInputMessage.content = "What is the weather there?";
    const result = buildOpenAIPayload("m", body, true, null);
    expect(result.messages).toHaveLength(5);
    expect(result.messages[0]).toEqual({ role: "user", content: "Hello, can you help me?" });
    expect(result.messages[1]).toEqual({ role: "assistant", content: "Of course! What do you need?" });
    expect(result.messages[2]).toEqual({ role: "user", content: "Tell me about Paris" });
    expect(result.messages[3]).toEqual({ role: "assistant", content: "Paris is the capital of France." });
    expect(result.messages[4]).toEqual({ role: "user", content: "What is the weather there?" });
  });

  it("converts assistant toolUses to tool_calls", () => {
    const body = makeKiroBody();
    body.conversationState.history = [
      { userInputMessage: { content: "What is the weather in Paris?", modelId: "claude-sonnet-4-6" } },
      {
        assistantResponseMessage: {
          content: "Let me check the weather.",
          toolUses: [{ toolUseId: "call_xxx", name: "get_weather", input: { city: "Paris" } }]
        }
      }
    ];
    body.conversationState.currentMessage.userInputMessage.content = "continue";
    const result = buildOpenAIPayload("m", body, true, null);
    expect(result.messages[1].role).toBe("assistant");
    expect(result.messages[1].tool_calls).toHaveLength(1);
    expect(result.messages[1].tool_calls[0]).toEqual({
      id: "call_xxx",
      type: "function",
      function: { name: "get_weather", arguments: '{"city":"Paris"}' }
    });
  });

  it("handles empty toolSpecification inputSchema", () => {
    const body = makeKiroBody();
    body.conversationState.currentMessage.userInputMessage.userInputMessageContext = {
      tools: [{
        toolSpecification: {
          name: "simple_tool",
          description: "A simple tool",
          inputSchema: { json: {} }
        }
      }]
    };
    const result = buildOpenAIPayload("m", body, true, null);
    expect(result.tools[0].function.parameters).toEqual({
      type: "object", properties: {}, required: []
    });
  });

  it("handles missing conversationState gracefully", () => {
    const result = buildOpenAIPayload("m", {}, true, null);
    expect(result.model).toBe("m");
    expect(result.messages).toHaveLength(0);
    expect(result.stream).toBe(true);
  });

  it("skips empty user content after stripping timestamp", () => {
    const body = makeKiroBody();
    body.conversationState.currentMessage.userInputMessage.content =
      "[Context: Current time is 2026-04-29T10:30:00.000Z]\n\n";
    const result = buildOpenAIPayload("m", body, true, null);
    expect(result.messages).toHaveLength(0);
  });

  it("defaults stream to true when not provided", () => {
    const result = buildOpenAIPayload("m", makeKiroBody(), undefined, null);
    expect(result.stream).toBe(true);
  });

  it("sets stream to false when explicitly false", () => {
    const result = buildOpenAIPayload("m", makeKiroBody(), false, null);
    expect(result.stream).toBe(false);
  });
});

// ── Response: openai → kiro (binary EventStream) ──────────

describe("convertOpenAIToKiro (openai → kiro EventStream)", () => {
  it("converts text content to assistantResponseEvent with modelId", () => {
    const chunk = makeOpenAIChunk({
      choice: { delta: { content: "Hello, how can I help?" } }
    });
    const state = initKiroState();
    const { eventType, payload } = decodeFirst(convertOpenAIToKiro(chunk, state));
    expect(eventType).toBe("assistantResponseEvent");
    expect(payload.content).toBe("Hello, how can I help?");
    expect(payload.modelId).toBe("gpt-5");
  });

  it("passes <thinking> tags through as text content (no extraction)", () => {
    const chunk = makeOpenAIChunk({
      choice: { delta: { content: "<thinking>Let me think...</thinking>Here is the answer" } }
    });
    const state = initKiroState();
    const { eventType, payload } = decodeFirst(convertOpenAIToKiro(chunk, state));
    // Text content is passed through as-is — reasoningContentEvent is only for delta.reasoning_content
    expect(eventType).toBe("assistantResponseEvent");
    expect(payload.content).toBe("<thinking>Let me think...</thinking>Here is the answer");
  });

  it("emits reasoningContentEvent for delta.reasoning_content", () => {
    const chunk = makeOpenAIChunk({
      choice: { delta: { reasoning_content: "Step-by-step analysis" } }
    });
    const state = initKiroState();
    const { eventType, payload } = decodeFirst(convertOpenAIToKiro(chunk, state));
    expect(eventType).toBe("reasoningContentEvent");
    expect(payload.content).toBe("Step-by-step analysis");
    expect(payload.modelId).toBe("gpt-5");
  });

  it("emits both reasoning and text when both delta fields present", () => {
    const chunk = makeOpenAIChunk({
      choice: {
        delta: { reasoning_content: "Let me think...", content: "Here is the answer" }
      }
    });
    const state = initKiroState();
    const decoded = decodeAll(convertOpenAIToKiro(chunk, state));
    expect(decoded).toHaveLength(2);
    expect(decoded[0].eventType).toBe("reasoningContentEvent");
    expect(decoded[0].payload.content).toBe("Let me think...");
    expect(decoded[1].eventType).toBe("assistantResponseEvent");
    expect(decoded[1].payload.content).toBe("Here is the answer");
  });

  it("streams tool calls incrementally with stop:true at finish", () => {
    const state = initKiroState();

    // First chunk: id + name → emits {name, toolUseId} frame (no input)
    let result = convertOpenAIToKiro(makeOpenAIChunk({
      choice: {
        delta: {
          tool_calls: [{
            index: 0, id: "call_abc", type: "function",
            function: { name: "get_weather", arguments: "" }
          }]
        }
      }
    }), state);
    const decoded1 = decodeAll(result);
    expect(decoded1).toHaveLength(1);
    expect(decoded1[0].eventType).toBe("toolUseEvent");
    expect(decoded1[0].payload.name).toBe("get_weather");
    expect(decoded1[0].payload.toolUseId).toBe("call_abc");
    expect(decoded1[0].payload.input).toBeUndefined(); // first frame has no input

    // Second chunk: arguments delta → emits {input: "<fragment>", ...}
    result = convertOpenAIToKiro(makeOpenAIChunk({
      choice: {
        delta: {
          tool_calls: [{ index: 0, function: { arguments: '{"city":"Paris"}' } }]
        }
      }
    }), state);
    const decoded2 = decodeAll(result);
    expect(decoded2).toHaveLength(1);
    expect(decoded2[0].eventType).toBe("toolUseEvent");
    expect(decoded2[0].payload.input).toBe('{"city":"Paris"}');

    // Finish: emits {name, stop: true, toolUseId} (NOT messageStopEvent)
    result = convertOpenAIToKiro(makeOpenAIChunk({
      choice: { delta: {}, finish_reason: "tool_calls" }
    }), state);
    const decoded3 = decodeAll(result);
    const stopFrame = decoded3.find(d => d.payload.stop === true);
    expect(stopFrame).toBeDefined();
    expect(stopFrame.eventType).toBe("toolUseEvent");
    expect(stopFrame.payload.toolUseId).toBe("call_abc");
    expect(decoded3.find(d => d.eventType === "messageStopEvent")).toBeUndefined();
  });

  it("emits messageStopEvent on finish_reason stop", () => {
    const chunk = makeOpenAIChunk({ choice: { delta: {}, finish_reason: "stop" } });
    const { eventType } = decodeFirst(convertOpenAIToKiro(chunk, initKiroState()));
    expect(eventType).toBe("messageStopEvent");
  });

  it("emits messageStopEvent on finish_reason length", () => {
    const chunk = makeOpenAIChunk({ choice: { delta: {}, finish_reason: "length" } });
    const { eventType } = decodeFirst(convertOpenAIToKiro(chunk, initKiroState()));
    expect(eventType).toBe("messageStopEvent");
  });

  it("emits usageEvent with correct token mapping", () => {
    const state = initKiroState();
    let result = convertOpenAIToKiro(makeOpenAIChunk({
      topLevel: { usage: { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 } }
    }), state);
    expect(result).toBeNull();

    result = convertOpenAIToKiro(makeOpenAIChunk({
      choice: { delta: {}, finish_reason: "stop" }
    }), state);

    const decoded = decodeAll(result);
    const usageEvent = decoded.find(d => d.eventType === "usageEvent");
    expect(usageEvent).toBeDefined();
    expect(usageEvent.payload.inputTokens).toBe(100);
    expect(usageEvent.payload.outputTokens).toBe(50);
  });

  it("skips empty content", () => {
    const chunk = makeOpenAIChunk({ choice: { delta: { content: "" } } });
    expect(convertOpenAIToKiro(chunk, initKiroState())).toBeNull();
  });

  it("streams multiple tool calls incrementally", () => {
    const state = initKiroState();

    // Tool 1 first appearance
    let decoded = decodeAll(convertOpenAIToKiro(makeOpenAIChunk({
      choice: {
        delta: {
          tool_calls: [{
            index: 0, id: "call_1", type: "function",
            function: { name: "get_weather", arguments: '{"city":"Paris"}' }
          }]
        }
      }
    }), state));
    // First frame: {name, toolUseId}, second frame: {input, name, toolUseId}
    expect(decoded).toHaveLength(2);
    expect(decoded[0].payload.name).toBe("get_weather");
    expect(decoded[0].payload.input).toBeUndefined();
    expect(decoded[1].payload.input).toBe('{"city":"Paris"}');

    // Tool 2 first appearance
    decoded = decodeAll(convertOpenAIToKiro(makeOpenAIChunk({
      choice: {
        delta: {
          tool_calls: [{
            index: 1, id: "call_2", type: "function",
            function: { name: "get_time", arguments: '{"timezone":"CET"}' }
          }]
        }
      }
    }), state));
    expect(decoded[0].payload.name).toBe("get_time");
    expect(decoded[0].payload.input).toBeUndefined();
    expect(decoded[1].payload.input).toBe('{"timezone":"CET"}');

    // Finish: two stop frames
    decoded = decodeAll(convertOpenAIToKiro(makeOpenAIChunk({
      choice: { delta: {}, finish_reason: "tool_calls" }
    }), state));
    const stops = decoded.filter(d => d.payload.stop === true);
    expect(stops).toHaveLength(2);
    expect(stops[0].payload.toolUseId).toBe("call_1");
    expect(stops[1].payload.toolUseId).toBe("call_2");
  });

  it("handles usage arriving with finish chunk", () => {
    const chunk = makeOpenAIChunk({
      choice: { delta: {}, finish_reason: "stop" },
      topLevel: { usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 } }
    });
    const decoded = decodeAll(convertOpenAIToKiro(chunk, initKiroState()));
    expect(decoded.find(d => d.eventType === "usageEvent")).toBeDefined();
    expect(decoded.find(d => d.eventType === "messageStopEvent")).toBeDefined();
  });

  it("emits messageStopEvent on flush when not previously sent", () => {
    const { eventType } = decodeFirst(convertOpenAIToKiro(null, initKiroState()));
    expect(eventType).toBe("messageStopEvent");
  });

  it("returns null on flush when finish already sent", () => {
    const state = initKiroState();
    state.finishSent = true;
    expect(convertOpenAIToKiro(null, state)).toBeNull();
  });

  it("passes tool call arguments through as raw string", () => {
    const state = initKiroState();
    // First appearance (id+name) and first input fragment arrive in same delta
    const decoded = decodeAll(convertOpenAIToKiro(makeOpenAIChunk({
      choice: {
        delta: {
          tool_calls: [{
            index: 0, id: "call_raw", type: "function",
            function: { name: "raw_tool", arguments: "not valid json" }
          }]
        }
      }
    }), state));
    // First frame: init (no input), second frame: input fragment
    expect(decoded[0].payload.name).toBe("raw_tool");
    expect(decoded[0].payload.input).toBeUndefined();
    expect(decoded[1].payload.input).toBe("not valid json");
  });

  it("handles chunk with no choices array", () => {
    const chunk = { id: "chatcmpl-test", object: "chat.completion.chunk", created: 1717000000, model: "gpt-5" };
    expect(convertOpenAIToKiro(chunk, initKiroState())).toBeNull();
  });

  it("includes :message-type header in every frame", () => {
    const chunk = makeOpenAIChunk({ choice: { delta: { content: "Hello" } } });
    const result = convertOpenAIToKiro(chunk, initKiroState());
    const frame = Array.isArray(result) ? result[0] : result;
    const msg = codec.decode(frame);
    expect(msg.headers[":message-type"].value).toBe("event");
    expect(msg.headers[":content-type"].value).toBe("application/json");
  });
});

// ── Round-trip: openai → kiro → openai ─────────────────────

describe("round-trip: openai → kiro → openai", () => {
  it("preserves messages, tools, and config through round-trip", () => {
    const openaiBody = {
      model: "gpt-5",
      messages: [
        { role: "user", content: "What is the weather in Paris?" },
        {
          role: "assistant", content: "Let me check.",
          tool_calls: [{
            id: "call_1", type: "function",
            function: { name: "get_weather", arguments: '{"city":"Paris"}' }
          }]
        },
        { role: "tool", tool_call_id: "call_1", content: "Paris: 22°C, sunny" },
        { role: "user", content: "Thanks!" }
      ],
      tools: [{
        type: "function",
        function: {
          name: "get_weather", description: "Get weather for a city",
          parameters: {
            type: "object", properties: { city: { type: "string" } }, required: ["city"]
          }
        }
      }],
      max_tokens: 8000, temperature: 0.5, top_p: 0.95, stream: true
    };

    const kiroPayload = buildKiroPayload(openaiBody.model, openaiBody, true, null);
    const result = buildOpenAIPayload(openaiBody.model, kiroPayload, true, null);

    expect(result.messages.length).toBeGreaterThanOrEqual(3);
    const roles = result.messages.map(m => m.role);
    expect(roles).toContain("user");
    expect(roles).toContain("assistant");
    expect(roles).toContain("tool");
    expect(result.tools).toHaveLength(1);
    expect(result.tools[0].function.name).toBe("get_weather");
    expect(result.max_tokens).toBe(8000);
    expect(result.temperature).toBe(0.5);
    expect(result.top_p).toBe(0.95);
    expect(result.model).toBe("gpt-5");
  });

  it("handles conversation with only user messages", () => {
    const openaiBody = {
      model: "gpt-5",
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi! How can I help?" },
        { role: "user", content: "Tell me a joke" }
      ],
      stream: true
    };
    const kiroPayload = buildKiroPayload(openaiBody.model, openaiBody, true, null);
    const result = buildOpenAIPayload(openaiBody.model, kiroPayload, true, null);
    expect(result.messages).toHaveLength(3);
    expect(result.messages[0].role).toBe("user");
    expect(result.messages[1].role).toBe("assistant");
    expect(result.messages[2].role).toBe("user");
  });
});

// ── Round-trip: kiro → openai → kiro ──────────────────────

describe("round-trip: kiro → openai → kiro", () => {
  it("preserves structural fidelity through round-trip", () => {
    const kiroBody = makeKiroBody();
    kiroBody.conversationState.history = [
      { userInputMessage: { content: "Hello", modelId: "claude-sonnet-4-6" } },
      { assistantResponseMessage: { content: "Hi there!" } }
    ];
    kiroBody.conversationState.currentMessage.userInputMessage.content = "How are you?";
    kiroBody.conversationState.currentMessage.userInputMessage.userInputMessageContext = {
      tools: [{
        toolSpecification: {
          name: "greet", description: "Greet the user",
          inputSchema: { json: { type: "object", properties: {}, required: [] } }
        }
      }]
    };
    const openaiBody = buildOpenAIPayload("claude-sonnet-4-6", kiroBody, true, null);
    const result = buildKiroPayload(openaiBody.model, openaiBody, true, null);
    expect(result.conversationState.currentMessage.userInputMessage).toBeDefined();
    expect(result.conversationState.currentMessage.userInputMessage.content).toContain("How are you?");
    expect(result.conversationState.history.length).toBeGreaterThanOrEqual(2);
  });

  it("preserves model name in modelId field", () => {
    const kiroBody = makeKiroBody();
    const openaiBody = buildOpenAIPayload("my-model", kiroBody, true, null);
    const result = buildKiroPayload("my-model", openaiBody, true, null);
    expect(result.conversationState.currentMessage.userInputMessage.modelId).toBe("my-model");
  });

  it("preserves inference config in round-trip", () => {
    const kiroBody = makeKiroBody();
    kiroBody.inferenceConfig = { maxTokens: 4000, temperature: 0.2, topP: 1.0 };
    const openaiBody = buildOpenAIPayload("m", kiroBody, true, null);
    expect(openaiBody.max_tokens).toBe(4000);
    expect(openaiBody.temperature).toBe(0.2);
    expect(openaiBody.top_p).toBe(1.0);
  });
});

// ── Response round-trip: kiro → openai → kiro (EventStream)

describe("response round-trip: kiro → openai → kiro (EventStream)", () => {
  function makeKiroSSE(eventType, data) {
    return `event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`;
  }

  it("round-trips assistantResponseEvent", () => {
    const kiroEvent = makeKiroSSE("assistantResponseEvent", {
      assistantResponseEvent: { content: "Hello world" }
    });
    const state1 = {};
    const openaiChunk = convertKiroToOpenAI(kiroEvent, state1);
    expect(openaiChunk.choices[0].delta.content).toBe("Hello world");

    const state2 = initKiroState();
    const { eventType, payload } = decodeFirst(convertOpenAIToKiro(openaiChunk, state2));
    expect(eventType).toBe("assistantResponseEvent");
    expect(payload.content).toBe("Hello world");
    expect(payload.modelId).toBeDefined();
  });

  it("round-trips reasoningContentEvent", () => {
    const kiroEvent = makeKiroSSE("reasoningContentEvent", {
      reasoningContentEvent: { content: "Let me think about this" }
    });
    const state1 = {};
    const openaiChunk = convertKiroToOpenAI(kiroEvent, state1);
    // kiro→openai wraps reasoning in <thinking> tags
    expect(openaiChunk.choices[0].delta.content).toBe("<thinking>Let me think about this</thinking>");

    // openai→kiro for text with <thinking> passes through as assistantResponseEvent
    // (only delta.reasoning_content triggers reasoningContentEvent)
    const state2 = initKiroState();
    const { eventType, payload } = decodeFirst(convertOpenAIToKiro(openaiChunk, state2));
    expect(eventType).toBe("assistantResponseEvent");
    expect(payload.content).toContain("<thinking>");
    expect(payload.content).toContain("Let me think about this");
  });

  it("round-trips messageStopEvent", () => {
    const kiroEvent = makeKiroSSE("messageStopEvent", { messageStopEvent: {} });
    const state1 = {};
    const openaiChunk = convertKiroToOpenAI(kiroEvent, state1);
    expect(openaiChunk.choices[0].finish_reason).toBe("stop");
    const { eventType } = decodeFirst(convertOpenAIToKiro(openaiChunk, initKiroState()));
    expect(eventType).toBe("messageStopEvent");
  });

  it("round-trips toolUseEvent through incremental streaming", () => {
    // Kiro toolUseEvent with complete input → becomes OpenAI tool_calls delta
    const kiroEvent = makeKiroSSE("toolUseEvent", {
      toolUseEvent: { toolUseId: "call_test", name: "get_weather", input: { city: "Paris" } }
    });
    const state1 = {};
    const openaiChunk = convertKiroToOpenAI(kiroEvent, state1);
    expect(openaiChunk.choices[0].delta.tool_calls[0].function.name).toBe("get_weather");

    // OpenAI delta → kiro incremental frames
    const state2 = initKiroState();
    // First: init frame (no input) + input frame (stringified args)
    const decodedInit = decodeAll(convertOpenAIToKiro(openaiChunk, state2));
    expect(decodedInit[0].eventType).toBe("toolUseEvent");
    expect(decodedInit[0].payload.name).toBe("get_weather");
    expect(decodedInit[0].payload.input).toBeUndefined();
    expect(decodedInit[1].payload.input).toBe('{"city":"Paris"}');

    // Finish: stop frame
    const decodedStop = decodeAll(convertOpenAIToKiro(makeOpenAIChunk({
      choice: { delta: {}, finish_reason: "tool_calls" }
    }), state2));
    expect(decodedStop[0].payload.stop).toBe(true);
    expect(decodedStop[0].payload.toolUseId).toBe("call_test");
  });
});

// ── Binary frame validation ────────────────────────────────

describe("EventStream binary frame output", () => {
  it("produces valid binary frames decodable by EventStreamCodec", () => {
    const chunk = makeOpenAIChunk({ choice: { delta: { content: "Hello" } } });
    const result = convertOpenAIToKiro(chunk, initKiroState());
    const frame = Array.isArray(result) ? result[0] : result;
    expect(frame).toBeInstanceOf(Uint8Array);
    const msg = codec.decode(frame);
    expect(msg.headers).toBeDefined();
  });

  it("produces frames with correct :event-type header", () => {
    const chunk = makeOpenAIChunk({ choice: { delta: { content: "Test" } } });
    const { eventType, payload } = decodeFirst(convertOpenAIToKiro(chunk, initKiroState()));
    expect(eventType).toBe("assistantResponseEvent");
    expect(payload).toBeDefined();
  });

  it("produces non-empty frames", () => {
    const chunk = makeOpenAIChunk({ choice: { delta: { content: "Hello" } } });
    const result = convertOpenAIToKiro(chunk, initKiroState());
    const frame = Array.isArray(result) ? result[0] : result;
    expect(frame.length).toBeGreaterThan(0);
  });

  it("correctly decodes the payload as JSON", () => {
    const chunk = makeOpenAIChunk({ choice: { delta: { content: "Hello world" } } });
    const result = convertOpenAIToKiro(chunk, initKiroState());
    const msg = codec.decode(Array.isArray(result) ? result[0] : result);
    const body = JSON.parse(toUtf8(msg.body));
    expect(body.content).toBe("Hello world");
  });
});

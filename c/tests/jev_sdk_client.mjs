// Drives TypeSafe's TypeScript SDK (@typesafe-ai/sdk), unmodified, for
// tests/test_jev_sdk.py: TYPESAFE_API_KEY and TYPESAFE_BASE_URL are the only
// change, read by the SDK itself. The request comes as argv[2]; the parsed
// reply, its request id and the model list go to stdout as JSON, or the
// SDK's error (name, status, message) when it raised one.
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(path.join(process.env.JEV_TS_SDK_DIR || process.cwd(), "package.json"));
const { TypeSafeClient, choice, noul, score } = require("@typesafe-ai/sdk");

const request = JSON.parse(process.argv[2]);
// The test's own settings: a CPU box can take longer than the 10 s default,
// and a failure should show once rather than after two retries.
const client = new TypeSafeClient({ timeout: 120000, retry: { maxRetries: 0 } });
const body = request.helpers
  ? {
      state: "I was charged twice. Please help.",
      questions: {
        tone: choice("What is the tone?", { calm: null, angry: null }),
        angry: noul("Is the sender angry?"),
        level: score("How strong?", ["weak", "strong"]),
      },
    }
  : { state: request.state, questions: request.questions };
try {
  const { data, requestId } = await client.systemOne(body).withResponse();
  const models = await client.models.list();
  process.stdout.write(JSON.stringify({ result: data, requestId, models }));
} catch (error) {
  process.stdout.write(JSON.stringify({ error: { name: error.name, status: error.status, message: error.message } }));
}

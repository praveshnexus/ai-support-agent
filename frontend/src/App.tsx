import { useState } from "react";
import {
  Bot,
  Plus,
  MessageSquare,
  Settings,
  Send,
  Headphones,
  Sparkles,
  ArrowUpRight,
  CircleHelp,
  Clock3,
  FileText,
  AlertCircle,
} from "lucide-react";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  durationMs?: number;
  sources?: string[];
  isError?: boolean;
};

type ChatResponse = {
  reply?: string;
  detail?: string;
  duration_ms?: number;
  sources?: string[];
};

const suggestions = [
  "How do I reset my password?",
  "Where can I check my order status?",
  "How do I contact customer support?",
];

const API_URL = "http://127.0.0.1:8000/api/chat";

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function sendMessage(text = input) {
    const content = text.trim();

    if (!content || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content,
    };

    setMessages((previous) => [...previous, userMessage]);
    setInput("");
    setIsLoading(true);

    const startTime = performance.now();

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: content }),
      });

      const data = (await response.json().catch(() => null)) as
        | ChatResponse
        | null;

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            `The server returned an error (${response.status}).`,
        );
      }

      if (!data?.reply?.trim()) {
        throw new Error(
          "The AI returned an empty response. Please try again.",
        );
      }

      const durationMs =
        typeof data.duration_ms === "number"
          ? data.duration_ms
          : Math.round(performance.now() - startTime);

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.reply,
        durationMs,
        sources: data.sources ?? [],
      };

      setMessages((previous) => [...previous, assistantMessage]);
    } catch (error) {
      console.error("Chat request failed:", error);

      let errorMessage = "Something went wrong. Please try again.";

      if (error instanceof Error) {
        errorMessage =
          error.message === "Failed to fetch"
            ? "I couldn't reach the backend. Check that FastAPI is running at http://127.0.0.1:8000."
            : error.message;
      }

      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: errorMessage,
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  function startNewConversation() {
    if (isLoading) return;

    setMessages([]);
    setInput("");
  }

  return (
    <div className="flex min-h-screen bg-[#f8f9fc] text-slate-800">
      <aside className="hidden w-64 shrink-0 flex-col border-r border-slate-200 bg-white p-5 md:flex">
        <div className="mb-10 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-600 text-white shadow-lg shadow-violet-200">
            <Bot size={23} />
          </div>

          <div>
            <h1 className="text-lg font-bold tracking-tight">ResolveAI</h1>
            <p className="text-xs text-slate-400">Support workspace</p>
          </div>
        </div>

        <button
          onClick={startNewConversation}
          disabled={isLoading}
          className="flex items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-3 text-sm font-semibold text-white shadow-md shadow-violet-200 transition hover:bg-violet-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Plus size={18} />
          New conversation
        </button>

        <p className="mb-3 mt-9 px-2 text-[11px] font-bold tracking-[0.16em] text-slate-400">
          WORKSPACE
        </p>

        <button className="flex items-center gap-3 rounded-xl bg-violet-50 px-3 py-3 text-left text-sm font-semibold text-violet-700">
          <MessageSquare size={18} />
          AI Support Chat
        </button>

        <button
          type="button"
          className="mt-2 flex items-center gap-3 rounded-xl px-3 py-3 text-left text-sm text-slate-500 transition hover:bg-slate-50 hover:text-slate-800"
          onClick={() =>
            window.alert(
              "The support center will be available when ticket management is implemented.",
            )
          }
        >
          <Headphones size={18} />
          Support center
        </button>

        <div className="mt-auto border-t border-slate-100 pt-4">
          <button
            type="button"
            className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-sm text-slate-500 hover:bg-slate-50"
            onClick={() =>
              window.alert(
                "Settings will be available when user accounts and preferences are implemented.",
              )
            }
          >
            <Settings size={18} />
            Settings
          </button>

          <div className="mt-3 flex items-center gap-3 rounded-xl bg-slate-50 p-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-violet-100 text-sm font-bold text-violet-700">
              R
            </div>

            <div>
              <p className="text-sm font-semibold">Your workspace</p>
              <p className="text-xs text-slate-400">Local development</p>
            </div>
          </div>
        </div>
      </aside>

      <main className="flex min-h-screen min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-slate-200 bg-white px-5 py-4 sm:px-8">
          <div>
            <h2 className="font-semibold text-slate-800">AI Support</h2>
            <p className="mt-1 text-xs text-slate-400">
              Your customer support assistant
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            API configured
          </div>
        </header>

        <section className="flex flex-1 flex-col px-4 py-8 sm:px-8">
          {messages.length === 0 ? (
            <div className="m-auto w-full max-w-2xl py-8">
              <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-violet-100 text-violet-600">
                <Sparkles size={30} />
              </div>

              <p className="mb-3 text-center text-xs font-bold uppercase tracking-[0.2em] text-violet-600">
                YOUR SUPPORT ASSISTANT
              </p>

              <h1 className="text-center text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
                How can we help you today?
              </h1>

              <p className="mx-auto mt-4 max-w-md text-center text-sm leading-6 text-slate-500">
                Ask us about your account, orders, or support questions.
                We&apos;re here to help.
              </p>

              <div className="mb-3 mt-10 flex items-center gap-2 text-xs font-semibold text-slate-400">
                <CircleHelp size={15} />
                TRY ASKING
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => void sendMessage(suggestion)}
                    disabled={isLoading}
                    className="group flex min-h-20 items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-white p-4 text-left text-sm font-medium text-slate-600 shadow-sm transition hover:border-violet-300 hover:bg-violet-50/50 hover:text-violet-700 disabled:opacity-50"
                  >
                    <span>{suggestion}</span>
                    <ArrowUpRight
                      size={17}
                      className="shrink-0 text-slate-400 transition group-hover:text-violet-600"
                    />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex items-start gap-3 ${
                    message.role === "user" ? "flex-row-reverse" : ""
                  }`}
                >
                  <div
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${
                      message.role === "assistant"
                        ? message.isError
                          ? "bg-red-100 text-red-700"
                          : "bg-violet-100 text-violet-700"
                        : "bg-slate-200 text-slate-700"
                    }`}
                  >
                    {message.role === "assistant" ? (
                      message.isError ? (
                        <AlertCircle size={19} />
                      ) : (
                        <Bot size={19} />
                      )
                    ) : (
                      <span className="text-sm font-bold">Y</span>
                    )}
                  </div>

                  <div className="flex max-w-[85%] flex-col gap-2 sm:max-w-[75%]">
                    <div
                      className={`whitespace-pre-wrap break-words rounded-2xl px-4 py-3 text-sm leading-7 ${
                        message.role === "user"
                          ? "rounded-tr-sm bg-violet-600 text-white"
                          : message.isError
                            ? "rounded-tl-sm border border-red-200 bg-red-50 text-red-700"
                            : "rounded-tl-sm border border-slate-200 bg-white text-slate-700 shadow-sm"
                      }`}
                    >
                      {message.content}
                    </div>

                    {message.role === "assistant" &&
                      message.sources &&
                      message.sources.length > 0 && (
                        <div className="rounded-xl border border-slate-200 bg-white p-3">
                          <p className="mb-2 flex items-center gap-2 text-xs font-semibold text-slate-500">
                            <FileText size={14} />
                            Knowledge sources
                          </p>

                          <div className="flex flex-wrap gap-2">
                            {message.sources.map((source) => (
                              <span
                                key={source}
                                className="max-w-full break-all rounded-lg bg-violet-50 px-2.5 py-1.5 text-xs text-violet-700"
                              >
                                {source}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                    {message.role === "assistant" &&
                      message.durationMs !== undefined && (
                        <p className="flex items-center gap-1.5 px-1 text-[11px] text-slate-400">
                          <Clock3 size={12} />
                          Responded in{" "}
                          {(message.durationMs / 1000).toFixed(1)}s
                        </p>
                      )}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex items-start gap-3">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-100 text-violet-700">
                    <Bot size={19} />
                  </div>

                  <div
                    role="status"
                    aria-live="polite"
                    className="flex items-center gap-3 rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3 text-sm text-slate-500 shadow-sm"
                  >
                    <span>Searching knowledge and generating a response</span>
                    <span className="flex gap-1">
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-violet-500 [animation-delay:-0.3s]" />
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-violet-500 [animation-delay:-0.15s]" />
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-violet-500" />
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        <div className="sticky bottom-0 bg-[#f8f9fc] px-4 pb-3 pt-2 sm:px-8">
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void sendMessage();
            }}
            className="mx-auto flex w-full max-w-3xl items-center gap-3 rounded-2xl border border-slate-200 bg-white p-2 shadow-sm transition focus-within:border-violet-300 focus-within:ring-4 focus-within:ring-violet-100/60"
          >
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Describe your issue..."
              aria-label="Your support message"
              disabled={isLoading}
              className="min-w-0 flex-1 bg-transparent px-3 py-3 text-sm text-slate-800 outline-none placeholder:text-slate-400 disabled:opacity-60"
            />

            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              aria-label="Send message"
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-violet-600 text-white transition hover:bg-violet-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isLoading ? (
                <span className="text-lg">…</span>
              ) : (
                <Send size={18} />
              )}
            </button>
          </form>

          <p className="mx-auto mt-3 max-w-3xl text-center text-[11px] text-slate-400">
            Powered by Gemini · FastAPI · Retrieval-Augmented Generation
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;
import { useState } from "react";
import { Check, Copy } from "lucide-react";

interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
  /** Allow long lines to wrap instead of scroll (for prompts/prose-like blocks) */
  wrap?: boolean;
}

const CodeBlock = ({ code, language = "bash", title, wrap = false }: CodeBlockProps) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="overflow-hidden rounded-lg border border-border/40 bg-[#0d0d14] shadow-2xl">
      <div className="flex items-center justify-between border-b border-white/5 px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-full bg-[#ff5f57]" />
          <div className="h-2.5 w-2.5 rounded-full bg-[#febc2e]" />
          <div className="h-2.5 w-2.5 rounded-full bg-[#28c840]" />
          {title && (
            <span className="ml-3 font-mono text-xs text-gray-500">{title}</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-gray-600">{language}</span>
          <button
            onClick={handleCopy}
            className="p-1.5 text-gray-500 transition-colors hover:text-gray-300"
            aria-label="Copy code"
          >
            {copied ? (
              <Check className="h-4 w-4 text-green-400" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>
      <pre className={`overflow-x-auto p-5 text-left text-sm leading-relaxed ${wrap ? "whitespace-pre-wrap break-words" : ""}`}>
        <code className="font-mono text-gray-300">{code}</code>
      </pre>
    </div>
  );
};

export default CodeBlock;

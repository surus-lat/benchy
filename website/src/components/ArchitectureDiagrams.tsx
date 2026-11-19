import { useState, useEffect, useLayoutEffect, useRef } from "react";
import diagramOverview from "@/assets/benchy-diagram-overview.svg";
import diagramOverviewDark from "@/assets/benchy-diagram-overview-dark.svg";
import diagramContract from "@/assets/benchy-diagram-contract.svg";
import diagramContractDark from "@/assets/benchy-diagram-contract-dark.svg";

const diagrams = [
  { id: 1, src: diagramOverview, srcDark: diagramOverviewDark, label: "Overview", caption: "The authoring and execution path" },
  { id: 2, src: diagramContract, srcDark: diagramContractDark, label: "The contract", caption: "B = (P, S, D), R = (B, AI) — the benchmark is separate from the AI-system" },
];

const ArchitectureDiagrams = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [isDark, setIsDark] = useState(() => document.documentElement.classList.contains("dark"));
  const tabsRef = useRef<HTMLDivElement>(null);
  const pendingScrollRef = useRef(false);

  const handleTabClick = (index: number) => {
    pendingScrollRef.current = true;
    setActiveTab(index);
  };

  // Listen for theme changes
  useEffect(() => {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.attributeName === "class") {
          setIsDark(document.documentElement.classList.contains("dark"));
        }
      });
    });

    observer.observe(document.documentElement, { attributes: true });
    return () => observer.disconnect();
  }, []);

  // Scroll after DOM updates to ensure consistent positioning
  useLayoutEffect(() => {
    if (!pendingScrollRef.current || !tabsRef.current) return;
    pendingScrollRef.current = false;

    const tabsRect = tabsRef.current.getBoundingClientRect();
    const tabsTopOnPage = tabsRect.top + window.scrollY;
    const header = document.querySelector("header");
    const headerHeight = header?.getBoundingClientRect().height ?? 0;
    const gap = 24; // breathing room below header
    const targetY = tabsTopOnPage - headerHeight - gap;

    window.scrollTo({ top: targetY, behavior: "smooth" });
  }, [activeTab]);

  // Preload all diagrams on mount (both light and dark)
  useEffect(() => {
    diagrams.forEach((diagram) => {
      const imgLight = new Image();
      imgLight.src = diagram.src;
      const imgDark = new Image();
      imgDark.src = diagram.srcDark;
    });
  }, []);

  return (
    <div className="space-y-6">
      {/* Pill-style tabs */}
      <div className="flex justify-center gap-2 scroll-mt-24" ref={tabsRef}>
        {diagrams.map((diagram, index) => (
          <button
            key={diagram.id}
            onClick={() => handleTabClick(index)}
            className={`rounded-full px-5 py-2 font-mono text-sm transition-all ${
              activeTab === index
                ? "bg-primary text-primary-foreground shadow-md"
                : "bg-muted text-muted-foreground hover:bg-muted/80 hover:text-foreground"
            }`}
          >
            {diagram.label}
          </button>
        ))}
      </div>

      {/* Diagram display - all rendered, visibility toggled */}
      <div className="p-4">
        <div className="flex justify-center">
          {diagrams.map((diagram, index) => (
            <img
              key={diagram.id}
              src={isDark ? diagram.srcDark : diagram.src}
              alt={`Benchy architecture - ${diagram.label}`}
              className={`max-w-full max-h-[65vh] w-auto h-auto ${activeTab !== index ? "hidden" : ""}`}
              style={{ filter: "drop-shadow(0 4px 12px rgba(0, 0, 0, 0.08))" }}
            />
          ))}
        </div>
        <p className="mt-6 text-center text-sm text-muted-foreground">
          {diagrams[activeTab].caption}
        </p>
      </div>
    </div>
  );
};

export default ArchitectureDiagrams;

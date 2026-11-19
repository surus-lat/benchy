import { Github, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

interface HeaderLinkProps {
  href: string;
  children: React.ReactNode;
  external?: boolean;
}

const HeaderLink = ({ href, children, external }: HeaderLinkProps) => {
  return (
    <a
      href={href}
      target={external ? "_blank" : undefined}
      rel={external ? "noopener noreferrer" : undefined}
      className="text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
    >
      {children}
    </a>
  );
};

const Header = () => {
  const [isDark, setIsDark] = useState(() => {
    // Check if dark mode is already set, default to light mode (false)
    return document.documentElement.classList.contains("dark");
  });

  useEffect(() => {
    // Ensure light mode is default on initial load
    if (!document.documentElement.classList.contains("dark") && !document.documentElement.classList.contains("light")) {
      document.documentElement.classList.remove("dark");
    }
  }, []);

  const toggleTheme = () => {
    document.documentElement.classList.toggle("dark");
    setIsDark(!isDark);
  };

  return (
    <header className="sticky top-0 z-50 px-6 pt-4">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between rounded-full bg-background/80 px-6 backdrop-blur-md border border-border/50 shadow-lg shadow-black/5 dark:shadow-black/20">
        <a href="/" className="font-mono text-2xl font-semibold tracking-tight text-foreground">
          benchy
        </a>
        <nav className="flex items-center gap-8">
          <HeaderLink href="https://arxiv.org/abs/2609.30550" external>Paper</HeaderLink>
          <HeaderLink href="https://latamboard.ai" external>LatamBoard</HeaderLink>
          <a
            href="https://github.com/surus-lat/benchy"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-primary"
            aria-label="GitHub repository"
          >
            <Github className="h-5 w-5" />
          </a>
          <button
            onClick={toggleTheme}
            className="rounded-full p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label="Toggle theme"
          >
            {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </nav>
      </div>
    </header>
  );
};

export default Header;

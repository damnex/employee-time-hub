import { Moon, SunMedium } from "lucide-react";
import { useTheme } from "next-themes";

import { Button } from "@/components/ui/button";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const nextTheme = resolvedTheme === "dark" ? "light" : "dark";

  return (
    <Button
      type="button"
      variant="outline"
      size="icon"
      className="h-11 w-11 rounded-2xl border-border/70 bg-background/90 text-foreground shadow-[0_10px_28px_rgba(15,23,42,0.08)] hover:bg-accent hover:text-accent-foreground [&_svg]:size-5"
      onClick={() => setTheme(nextTheme)}
    >
      {resolvedTheme === "dark" ? <SunMedium /> : <Moon />}
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
}

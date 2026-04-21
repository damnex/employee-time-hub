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
      className="h-10 w-10 rounded-xl"
      onClick={() => setTheme(nextTheme)}
    >
      {resolvedTheme === "dark" ? <SunMedium /> : <Moon />}
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
}

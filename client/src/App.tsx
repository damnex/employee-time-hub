import { Switch, Route, useLocation } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";
import { Button } from "@/components/ui/button";
import { ScanLine } from "lucide-react";

// Pages
import Dashboard from "./pages/dashboard";
import Employees from "./pages/employees";
import Attendance from "./pages/attendance";
import GateSimulator from "./pages/gate";
import ReaderControl from "./pages/ReaderControl";
import NotFound from "./pages/not-found";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard}/>
      <Route path="/employees" component={Employees}/>
      <Route path="/attendance" component={Attendance}/>
      <Route path="/gate" component={GateSimulator}/>
      <Route path="/reader-control" component={ReaderControl}/>
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  const [location] = useLocation();
  const isGateRoute = location === "/gate";
  const sidebarStyle = {
    "--sidebar-width": "15rem",
    "--sidebar-width-icon": "4.375rem",
  } as React.CSSProperties;

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider>
          <SidebarProvider style={sidebarStyle}>
            <div className="flex h-screen w-full overflow-hidden bg-background">
              <AppSidebar />
              <div className="flex min-w-0 flex-1 flex-col">
                <header className="flex h-16 items-center gap-4 border-b border-border/60 bg-background/92 px-6 shadow-sm backdrop-blur shrink-0 z-10">
                  <SidebarTrigger className="-ml-1 h-11 w-11 rounded-2xl bg-background/90 shadow-[0_10px_28px_rgba(15,23,42,0.08)]" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-foreground">Syncronize</p>
                    <p className="text-xs text-muted-foreground">Access Control Platform</p>
                  </div>
                  {isGateRoute && (
                    <Button
                      type="button"
                      onClick={() => window.dispatchEvent(new CustomEvent("gate:open-manual-trigger"))}
                      className="h-10 rounded-xl bg-primary px-4 text-sm font-semibold text-primary-foreground shadow-sm hover:bg-primary/90"
                    >
                      <ScanLine className="mr-2 size-4" />
                      Manual Trigger
                    </Button>
                  )}
                  <ThemeToggle />
                </header>
                <main
                  className={`flex-1 bg-background ${isGateRoute ? "overflow-hidden" : "overflow-auto"}`}
                >
                  <Router />
                </main>
              </div>
            </div>
          </SidebarProvider>
          <Toaster />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;

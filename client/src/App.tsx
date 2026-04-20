import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";

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
  const sidebarStyle = {
    "--sidebar-width": "16rem",
    "--sidebar-width-icon": "4rem",
  } as React.CSSProperties;

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <SidebarProvider style={sidebarStyle}>
          <div className="flex h-screen w-full bg-background overflow-hidden">
            <AppSidebar />
            <div className="flex flex-col flex-1 min-w-0">
              <header className="flex h-14 items-center gap-4 border-b border-border/40 bg-background/95 backdrop-blur px-6 shrink-0 shadow-sm z-10">
                <SidebarTrigger className="-ml-2 hover-elevate active-elevate-2 p-2 rounded-md" />
                <div className="w-full flex justify-end">
                  {/* Future placement for user profile or theme toggle */}
                </div>
              </header>
              <main className="flex-1 overflow-auto bg-slate-50/50 dark:bg-transparent">
                <Router />
              </main>
            </div>
          </div>
        </SidebarProvider>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;

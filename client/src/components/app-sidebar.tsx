import { Link, useLocation } from "wouter";
import { LayoutDashboard, Users, Clock, ScanFace, Building2, Settings2 } from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
} from "@/components/ui/sidebar";

const items = [
  { title: "Dashboard", url: "/", icon: LayoutDashboard },
  { title: "Employees", url: "/employees", icon: Users },
  { title: "Attendance", url: "/attendance", icon: Clock },
  { title: "Gate Simulator", url: "/gate", icon: ScanFace },
  { title: "Reader Control", url: "/reader-control", icon: Settings2 },
];

export function AppSidebar() {
  const [location] = useLocation();

  return (
    <Sidebar collapsible="icon" className="border-r border-sidebar-border/70 bg-sidebar/95 backdrop-blur">
      <SidebarHeader className="flex flex-row items-center gap-3 px-4 py-5 transition-[padding] duration-300 ease-in-out group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:px-3">
        <div className="flex aspect-square size-10 shrink-0 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-sm">
          <Building2 className="size-4" />
        </div>
        <div className="flex min-w-0 flex-col gap-1 leading-none transition-opacity duration-200 group-data-[collapsible=icon]:hidden">
          <span className="font-semibold tracking-tight text-sidebar-foreground">Syncronize</span>
          <span className="text-xs text-sidebar-foreground/60">Access Control Platform</span>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup className="px-3 py-2">
          <SidebarGroupLabel className="px-3 text-[11px] font-semibold uppercase tracking-[0.2em] text-sidebar-foreground/55">
            Workspace
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton 
                    asChild 
                    isActive={location === item.url}
                    tooltip={item.title}
                    className="h-11 rounded-xl px-3 font-medium transition-all duration-300 ease-in-out group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:px-2"
                  >
                    <Link href={item.url} className="flex items-center gap-3">
                      <item.icon className="size-4 shrink-0" />
                      <span className="group-data-[collapsible=icon]:hidden">{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}

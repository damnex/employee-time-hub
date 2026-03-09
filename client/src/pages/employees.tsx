import { useState } from "react";
import { useEmployees, useCreateEmployee } from "@/hooks/use-employees";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogFooter, DialogDescription } from "@/components/ui/dialog";
import { Plus, ScanFace, CheckCircle2, UserCircle } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { insertEmployeeSchema } from "@shared/schema";

const formSchema = insertEmployeeSchema.extend({
  faceDescriptor: z.array(z.number()).nullable().optional()
});

export default function Employees() {
  const { data: employees, isLoading } = useEmployees();
  const createEmployee = useCreateEmployee();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [faceCaptured, setFaceCaptured] = useState(false);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      employeeCode: "",
      name: "",
      department: "",
      phone: "",
      email: "",
      rfidUid: "",
      isActive: true,
      faceDescriptor: null
    }
  });

  const handleCaptureFace = () => {
    // Simulate capturing a 128-d face embedding
    const fakeDescriptor = Array.from({ length: 128 }, () => Number(Math.random().toFixed(4)));
    form.setValue("faceDescriptor", fakeDescriptor);
    setFaceCaptured(true);
  };

  const onSubmit = (values: z.infer<typeof formSchema>) => {
    // Convert null to undefined for the API if necessary, but schema accepts it
    createEmployee.mutate(values as any, {
      onSuccess: () => {
        setIsDialogOpen(false);
        form.reset();
        setFaceCaptured(false);
      }
    });
  };

  return (
    <div className="p-6 md:p-8 space-y-6 animate-in fade-in duration-500">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Directory</h1>
          <p className="text-muted-foreground mt-1">Manage personnel, credentials, and biometric data.</p>
        </div>

        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button className="shadow-sm hover:-translate-y-0.5 transition-transform">
              <Plus className="size-4 mr-2" /> Add Employee
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[500px]">
            <DialogHeader>
              <DialogTitle>Register New Employee</DialogTitle>
              <DialogDescription>
                Enter employee details and provision access credentials below.
              </DialogDescription>
            </DialogHeader>

            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4 py-4">
                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="name"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Full Name</FormLabel>
                        <FormControl><Input placeholder="Jane Doe" {...field} /></FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="employeeCode"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Employee Code</FormLabel>
                        <FormControl><Input placeholder="EMP-1042" {...field} /></FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="department"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Department</FormLabel>
                        <FormControl><Input placeholder="Engineering" {...field} /></FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="email"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Email (Optional)</FormLabel>
                        <FormControl><Input type="email" placeholder="jane@company.com" {...field} value={field.value || ''} /></FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="border-t pt-4 mt-4">
                  <h4 className="text-sm font-semibold mb-3">Access Credentials</h4>
                  <div className="space-y-4">
                    <FormField
                      control={form.control}
                      name="rfidUid"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>RFID UID</FormLabel>
                          <FormControl><Input placeholder="A1B2C3D4" className="font-mono" {...field} /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    
                    <div className="space-y-2">
                      <Label>Biometric Profile (Face)</Label>
                      <div className="flex items-center gap-4">
                        <Button 
                          type="button" 
                          variant={faceCaptured ? "outline" : "secondary"} 
                          onClick={handleCaptureFace}
                          className={`w-full ${faceCaptured ? "border-emerald-500 text-emerald-600" : ""}`}
                        >
                          {faceCaptured ? (
                            <><CheckCircle2 className="size-4 mr-2" /> Profile Enrolled</>
                          ) : (
                            <><ScanFace className="size-4 mr-2" /> Capture Face Data</>
                          )}
                        </Button>
                      </div>
                      {!faceCaptured && <p className="text-xs text-muted-foreground">Required for 2FA gate verification.</p>}
                    </div>
                  </div>
                </div>

                <DialogFooter className="pt-4">
                  <Button type="button" variant="ghost" onClick={() => setIsDialogOpen(false)}>Cancel</Button>
                  <Button type="submit" disabled={createEmployee.isPending}>
                    {createEmployee.isPending ? "Saving..." : "Save Employee"}
                  </Button>
                </DialogFooter>
              </form>
            </Form>
          </DialogContent>
        </Dialog>
      </div>

      <Card className="border-border/50 shadow-sm overflow-hidden">
        <CardContent className="p-0">
          <Table>
            <TableHeader className="bg-muted/50">
              <TableRow>
                <TableHead className="pl-6">Employee</TableHead>
                <TableHead>Code</TableHead>
                <TableHead>Department</TableHead>
                <TableHead>RFID Badge</TableHead>
                <TableHead>Biometrics</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <TableRow key={i}>
                    <TableCell className="pl-6"><Skeleton className="h-5 w-32" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-20" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-24" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-24" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-16" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-16" /></TableCell>
                  </TableRow>
                ))
              ) : employees?.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-12 text-muted-foreground">
                    <UserCircle className="size-12 mx-auto mb-3 opacity-20" />
                    No employees found. Add one to get started.
                  </TableCell>
                </TableRow>
              ) : (
                employees?.map((emp) => (
                  <TableRow key={emp.id} className="hover:bg-muted/30">
                    <TableCell className="pl-6 font-medium text-foreground">{emp.name}</TableCell>
                    <TableCell className="text-muted-foreground">{emp.employeeCode}</TableCell>
                    <TableCell>{emp.department}</TableCell>
                    <TableCell><Badge variant="outline" className="font-mono bg-background">{emp.rfidUid}</Badge></TableCell>
                    <TableCell>
                      {emp.faceDescriptor ? (
                        <Badge variant="secondary" className="bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border-none">Enrolled</Badge>
                      ) : (
                        <span className="text-xs text-muted-foreground">Pending</span>
                      )}
                    </TableCell>
                    <TableCell>
                      {emp.isActive ? (
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                          <span className="text-sm">Active</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full bg-muted-foreground"></div>
                          <span className="text-sm text-muted-foreground">Inactive</span>
                        </div>
                      )}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

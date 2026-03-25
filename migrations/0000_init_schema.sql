CREATE TABLE "attendances" (
	"id" serial PRIMARY KEY NOT NULL,
	"employee_id" integer NOT NULL,
	"date" text NOT NULL,
	"entry_time" timestamp,
	"exit_time" timestamp,
	"working_hours" double precision,
	"verification_status" text NOT NULL,
	"device_id" text NOT NULL
);
--> statement-breakpoint
CREATE TABLE "devices" (
	"id" serial PRIMARY KEY NOT NULL,
	"device_id" text NOT NULL,
	"location" text NOT NULL,
	"device_type" text NOT NULL,
	CONSTRAINT "devices_device_id_unique" UNIQUE("device_id")
);
--> statement-breakpoint
CREATE TABLE "employees" (
	"id" serial PRIMARY KEY NOT NULL,
	"employee_code" text NOT NULL,
	"name" text NOT NULL,
	"department" text NOT NULL,
	"phone" text,
	"email" text,
	"rfid_uid" text NOT NULL,
	"face_descriptor" jsonb,
	"is_active" boolean DEFAULT true,
	"created_at" timestamp DEFAULT now(),
	CONSTRAINT "employees_employee_code_unique" UNIQUE("employee_code"),
	CONSTRAINT "employees_rfid_uid_unique" UNIQUE("rfid_uid")
);
--> statement-breakpoint
CREATE TABLE "gate_events" (
	"id" serial PRIMARY KEY NOT NULL,
	"employee_id" integer,
	"date" text NOT NULL,
	"occurred_at" timestamp DEFAULT now(),
	"rfid_uid" text NOT NULL,
	"device_id" text NOT NULL,
	"scan_technology" text DEFAULT 'HF_RFID' NOT NULL,
	"decision" text NOT NULL,
	"verification_status" text NOT NULL,
	"event_message" text NOT NULL,
	"movement_direction" text,
	"movement_axis" text,
	"movement_confidence" double precision,
	"match_confidence" double precision,
	"face_quality" double precision,
	"face_consistency" double precision,
	"face_capture_mode" text
);
--> statement-breakpoint
ALTER TABLE "attendances" ADD CONSTRAINT "attendances_employee_id_employees_id_fk" FOREIGN KEY ("employee_id") REFERENCES "public"."employees"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "gate_events" ADD CONSTRAINT "gate_events_employee_id_employees_id_fk" FOREIGN KEY ("employee_id") REFERENCES "public"."employees"("id") ON DELETE no action ON UPDATE no action;
import { defineConfig } from "drizzle-kit";
import { getDatabaseUrl, loadEnvironment, normalizeConnectionString, shouldEnableSsl } from "./server/env";

loadEnvironment();

const databaseUrl = getDatabaseUrl();

if (!databaseUrl) {
  throw new Error("DATABASE_URL, ensure the database is provisioned");
}

export default defineConfig({
  out: "./migrations",
  schema: "./shared/schema.ts",
  dialect: "postgresql",
  dbCredentials: {
    url: normalizeConnectionString(databaseUrl),
    ...(shouldEnableSsl(databaseUrl) ? { ssl: "require" as const } : {}),
  },
});

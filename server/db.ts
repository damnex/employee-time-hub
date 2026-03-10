import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "@shared/schema";
import { getDatabaseUrl, loadEnvironment, normalizeConnectionString, shouldEnableSsl } from "./env";

const { Pool } = pg;

loadEnvironment();

const connectionString = getDatabaseUrl();

export const pool = connectionString
  ? new Pool({
      connectionString: normalizeConnectionString(connectionString),
      ssl: shouldEnableSsl(connectionString) ? { rejectUnauthorized: false } : undefined,
      max: 10,
      idleTimeoutMillis: 30_000,
      connectionTimeoutMillis: 10_000,
    })
  : null;

export const db = pool ? drizzle(pool, { schema }) : null;

if (pool) {
  pool.on("error", (error) => {
    console.error("[db] Unexpected PostgreSQL pool error:", error);
  });
}

export async function verifyDatabaseConnection() {
  if (!pool) {
    return false;
  }

  await pool.query("select 1");
  return true;
}

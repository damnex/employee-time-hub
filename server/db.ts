import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "@shared/schema";
import { getDatabaseUrl, loadEnvironment } from "./env";

const { Pool } = pg;

loadEnvironment();

const connectionString = getDatabaseUrl();

function shouldEnableSsl(databaseUrl: string) {
  const sslMode = process.env.PGSSLMODE?.toLowerCase();
  if (sslMode && sslMode !== "disable") {
    return true;
  }

  try {
    const parsedUrl = new URL(databaseUrl);
    const urlSslMode = parsedUrl.searchParams.get("sslmode")?.toLowerCase();

    if (urlSslMode && urlSslMode !== "disable") {
      return true;
    }

    return parsedUrl.hostname.includes("supabase");
  } catch {
    return false;
  }
}

export const pool = connectionString
  ? new Pool({
      connectionString,
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

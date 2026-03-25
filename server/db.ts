import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "@shared/schema";
import { getDatabaseUrl, loadEnvironment, normalizeConnectionString, shouldEnableSsl } from "./env";

const { Pool } = pg;

loadEnvironment();

const connectionString = getDatabaseUrl();
let databaseReady = false;
let databaseEverConnected = false;

export const pool = connectionString
  ? new Pool({
      connectionString: normalizeConnectionString(connectionString),
      ssl: shouldEnableSsl(connectionString) ? { rejectUnauthorized: false } : undefined,
      max: 10,
      keepAlive: true,
      keepAliveInitialDelayMillis: 10_000,
      idleTimeoutMillis: 30_000,
      connectionTimeoutMillis: 10_000,
    })
  : null;

export const db = pool ? drizzle(pool, { schema }) : null;

if (pool) {
  pool.on("error", (error) => {
    console.error("[db] Unexpected PostgreSQL pool error:", error);
    databaseReady = false;
  });
}

export async function verifyDatabaseConnection() {
  if (!pool) {
    databaseReady = false;
    return false;
  }

  try {
    await pool.query("select 1");
    databaseReady = true;
    databaseEverConnected = true;
    return true;
  } catch (error) {
    databaseReady = false;
    console.warn("[db] PostgreSQL connection check failed. Falling back to in-memory storage.", error);
    return false;
  }
}

export function isDatabaseReady() {
  return databaseReady;
}

export function shouldUseDatabaseStorage() {
  return Boolean(db) && (databaseReady || databaseEverConnected);
}

-- scripts/databricks_grants.sql
-- Production Read-Only Least-Privilege Grants for AI Data Analyst Service Principal
-- Execute this script as an Account Admin or Metastore Admin in Databricks SQL.

-- 1. Grant USE CATALOG on workspace catalog
GRANT USE CATALOG ON CATALOG `workspace` TO `analyst_service_principal`;

-- 2. Grant USE SCHEMA on zomato_gold only (denying access to silver, bronze, raw, or default)
GRANT USE SCHEMA ON SCHEMA `workspace`.`zomato_gold` TO `analyst_service_principal`;

-- 3. Grant SELECT on all existing and future tables in zomato_gold
GRANT SELECT ON SCHEMA `workspace`.`zomato_gold` TO `analyst_service_principal`;

-- 4. Explicitly ensure NO write or DDL privileges are granted (Defense in Depth)
-- Ensure NO MODIFY, CREATE, DROP, ALTER, or ALL PRIVILEGES exist for this principal
REVOKE MODIFY ON SCHEMA `workspace`.`zomato_gold` FROM `analyst_service_principal`;
REVOKE CREATE ON SCHEMA `workspace`.`zomato_gold` FROM `analyst_service_principal`;
REVOKE APPLY TAG ON SCHEMA `workspace`.`zomato_gold` FROM `analyst_service_principal`;

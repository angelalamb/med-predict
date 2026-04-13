// =============================================================================
// Neo4j post-pipeline sanity checks
// Run these in the Neo4j AuraDB browser after each pipeline run.
// =============================================================================

// 1. Node counts per category
MATCH (d:Device)
RETURN d.category AS category, count(d) AS count
ORDER BY count DESC;

// 2. Nodes missing category (should be 0 after re-running ultrasound)
MATCH (d:Device)
WHERE d.category IS NULL OR d.category = ""
RETURN count(d) AS missing_category;

// 3. Edge count per category
MATCH (a:Device)-[:PREDICATED_ON]->(b:Device)
RETURN a.category AS category, count(*) AS edges
ORDER BY edges DESC;

// 4. Spot-check wearables — confirm excluded terms didn't slip through
MATCH (d:Device {category: "wearables"})
WHERE toLower(d.device_name) CONTAINS "fingertip"
   OR toLower(d.device_name) CONTAINS "handheld"
   OR toLower(d.device_name) CONTAINS "bedside"
   OR toLower(d.device_name) CONTAINS "portable"
   OR toLower(d.device_name) CONTAINS "spot check"
RETURN d.k_number, d.device_name
LIMIT 20;

// 5. Sample 5 devices from each new category
MATCH (d:Device)
WHERE d.category IN ["ai_ml_radiology", "wearables"]
RETURN d.category, d.k_number, d.device_name
ORDER BY d.category, d.k_number
LIMIT 10;

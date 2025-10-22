#!/usr/bin/env node

/**
 * Build script that embeds version information into the UI build
 * This ensures the UI always knows its version and build details
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Get package version
const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
const version = packageJson.version;

// Get git information
let gitCommit = 'unknown';
let buildDate = new Date().toISOString().split('T')[0];

try {
  gitCommit = execSync('git rev-parse --short HEAD', { encoding: 'utf8' }).trim();
  buildDate = new Date().toISOString().split('T')[0];
} catch (error) {
  console.warn('Could not get git information:', error.message);
}

// Create version info file
const versionInfo = {
  ui_version: version,
  build_date: buildDate,
  git_commit: gitCommit,
  environment: process.env.NODE_ENV || 'production',
  build_timestamp: new Date().toISOString()
};

const versionFile = path.join(__dirname, 'src', 'version.json');
fs.writeFileSync(versionFile, JSON.stringify(versionInfo, null, 2));

console.log('📦 Version info generated:');
console.log(`   UI Version: ${version}`);
console.log(`   Build Date: ${buildDate}`);
console.log(`   Git Commit: ${gitCommit}`);
console.log(`   File: ${versionFile}`);

// Also create a TypeScript version file for type safety
const tsVersionFile = path.join(__dirname, 'src', 'version.ts');
const tsContent = `// Auto-generated version file - do not edit manually
export const VERSION_INFO = ${JSON.stringify(versionInfo, null, 2)} as const;

export type VersionInfo = typeof VERSION_INFO;
`;

fs.writeFileSync(tsVersionFile, tsContent);
console.log(`   TypeScript file: ${tsVersionFile}`);
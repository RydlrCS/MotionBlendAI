/**
 * FileControlPanel - Right-side control panel with search functionality
 * 
 * Provides file management controls and search capabilities:
 * - Search icon that opens ElasticSearch popup
 * - File browser and management tools
 * - Motion file metadata display
 * - Integration with BlendMixer
 */
import React, {useState} from 'react'
import ElasticSearchPopup from './ElasticSearchPopup'

interface FileControlPanelProps {
  onFileSelect?: (file: any) => void
  selectedFiles?: any[]
}

export default function FileControlPanel({onFileSelect, selectedFiles = []}: FileControlPanelProps) {
  const [isSearchOpen, setIsSearchOpen] = useState(false)

  const handleSearchClick = () => {
    setIsSearchOpen(true)
  }

  const handleFileFromSearch = (file: any) => {
    onFileSelect?.(file)
    setIsSearchOpen(false)
  }

  return (
    <div className="file-control-panel">
      {/* Panel Header */}
      <div className="panel-header">
        <h3>File Controls</h3>
        <button 
          className="search-button"
          onClick={handleSearchClick}
          title="Search motion files using Elasticsearch"
        >
          🔍
        </button>
      </div>

      {/* File Browser Section */}
      <div className="file-browser-section">
        <h4>Motion Library</h4>
        <div className="file-stats">
          <div className="stat-item">
            <span className="stat-label">Available:</span>
            <span className="stat-value">97 files</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Selected:</span>
            <span className="stat-value">{selectedFiles.length}</span>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="quick-actions">
        <button className="action-button" title="Refresh file list">
          🔄 Refresh
        </button>
        <button className="action-button" title="Import new files">
          📁 Import
        </button>
        <button className="action-button" title="Export selected">
          💾 Export
        </button>
      </div>

      {/* Selected Files Display */}
      {selectedFiles.length > 0 && (
        <div className="selected-files">
          <h4>Selected Files</h4>
          <div className="selected-list">
            {selectedFiles.map((file, index) => (
              <div key={index} className="selected-file-item">
                <span className="file-name">{file.name}</span>
                <span className="file-format">{file.metadata?.format || 'Unknown'}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="panel-instructions">
        <h4>Instructions</h4>
        <ul>
          <li>Click 🔍 to search motion files</li>
          <li>Drag search results to BlendMixer</li>
          <li>Use quick actions to manage files</li>
          <li>Monitor selected files below</li>
        </ul>
      </div>

      {/* ElasticSearch Popup */}
      <ElasticSearchPopup 
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        onFileSelect={handleFileFromSearch}
      />
    </div>
  )
}
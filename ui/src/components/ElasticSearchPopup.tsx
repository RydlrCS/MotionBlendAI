/**
 * ElasticSearchPopup - Dynamic Search Interface
 * 
 * A popup modal for searching motion files using Elasticsearch vector search.
 * Features:
 * - Real-time vector similarity search
 * - Draggable results to BlendMixer
 * - Professional modal interface
 * - Metadata display for search results
 */
import React, {useState, useEffect} from 'react'
import {searchMotions} from '../client'

interface SearchResult {
  id: string
  name: string
  vector: number[]
  metadata: {
    frames: number
    joints: number
    duration: number
    format: string
  }
  similarity_score: number
}

interface ElasticSearchPopupProps {
  isOpen: boolean
  onClose: () => void
  onFileSelect?: (file: SearchResult) => void
}

/**
 * Draggable search result item with motion metadata
 */
interface DraggableResultProps {
  result: SearchResult
  onSelect: (file: SearchResult) => void
}

function DraggableResult({result, onSelect}: DraggableResultProps) {
  const handleDragStart = (e: React.DragEvent) => {
    // Set drag data for drop handling
    e.dataTransfer.setData('application/json', JSON.stringify({
      type: 'motion-file',
      data: result
    }))
    e.dataTransfer.effectAllowed = 'copy'
  }

  const handleClick = () => {
    onSelect(result)
  }

  return (
    <div 
      className="search-result-item"
      draggable
      onDragStart={handleDragStart}
      onClick={handleClick}
      title={`Drag to BlendMixer or click to select\nSimilarity: ${(result.similarity_score * 100).toFixed(1)}%`}
    >
      <div className="result-header">
        <span className="result-name">{result.name}</span>
        <span className="similarity-score">
          {(result.similarity_score * 100).toFixed(1)}%
        </span>
      </div>
      
      <div className="result-metadata">
        <div className="metadata-row">
          <span>Format: {result.metadata.format}</span>
          <span>Frames: {result.metadata.frames}</span>
        </div>
        <div className="metadata-row">
          <span>Joints: {result.metadata.joints}</span>
          <span>Duration: {result.metadata.duration.toFixed(2)}s</span>
        </div>
      </div>
      
      <div className="drag-indicator">
        <span>⋮⋮</span> Drag to mixer
      </div>
    </div>
  )
}

/**
 * Main ElasticSearch popup component
 */
export default function ElasticSearchPopup({isOpen, onClose, onFileSelect}: ElasticSearchPopupProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [searchMode, setSearchMode] = useState<'text'|'vector'>('text')
  const [vectorInput, setVectorInput] = useState('')

  // Close popup on Escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }

    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      return () => document.removeEventListener('keydown', handleEscape)
    }
  }, [isOpen, onClose])

  /**
   * Perform text-based motion search
   */
  const performTextSearch = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([])
      return
    }

    setIsSearching(true)
    try {
      // Mock search results for development
      // In production, this would call the actual Elasticsearch API
      const mockResults: SearchResult[] = [
        {
          id: 'motion_001',
          name: 'Walking Forward',
          vector: [0.1, 0.2, 0.3, 0.4],
          metadata: {
            frames: 120,
            joints: 25,
            duration: 4.0,
            format: 'FBX'
          },
          similarity_score: 0.95
        },
        {
          id: 'motion_002', 
          name: 'Running Sprint',
          vector: [0.2, 0.3, 0.1, 0.5],
          metadata: {
            frames: 90,
            joints: 25,
            duration: 3.0,
            format: 'GLB'
          },
          similarity_score: 0.87
        },
        {
          id: 'motion_003',
          name: 'Dance Moves',
          vector: [0.5, 0.1, 0.4, 0.2],
          metadata: {
            frames: 200,
            joints: 30,
            duration: 6.7,
            format: 'TRC'
          },
          similarity_score: 0.72
        }
      ].filter(result => 
        result.name.toLowerCase().includes(query.toLowerCase())
      )

      setSearchResults(mockResults)
    } catch (error) {
      console.error('Search failed:', error)
      setSearchResults([])
    } finally {
      setIsSearching(false)
    }
  }

  /**
   * Perform vector-based similarity search
   */
  const performVectorSearch = async (vector: number[]) => {
    setIsSearching(true)
    try {
      const response = await searchMotions({
        vector: vector,
        k: 10
      })
      setSearchResults(response || [])
    } catch (error) {
      console.error('Vector search failed:', error)
      setSearchResults([])
    } finally {
      setIsSearching(false)
    }
  }

  /**
   * Handle search input changes with debouncing
   */
  useEffect(() => {
    if (searchMode === 'text') {
      const timeoutId = setTimeout(() => {
        performTextSearch(searchQuery)
      }, 300)
      return () => clearTimeout(timeoutId)
    }
  }, [searchQuery, searchMode])

  /**
   * Handle vector search submission
   */
  const handleVectorSearch = () => {
    try {
      const vector = JSON.parse(vectorInput)
      if (Array.isArray(vector) && vector.every(v => typeof v === 'number')) {
        performVectorSearch(vector)
      } else {
        alert('Please enter a valid array of numbers')
      }
    } catch (error) {
      alert('Invalid JSON format for vector')
    }
  }

  /**
   * Handle file selection from search results
   */
  const handleFileSelect = (file: SearchResult) => {
    onFileSelect?.(file)
    console.log('Selected motion file:', file.name)
  }

  if (!isOpen) return null

  return (
    <div className="elastic-search-overlay">
      <div className="elastic-search-popup">
        {/* Header with close button */}
        <div className="popup-header">
          <h3>🔍 Motion Search</h3>
          <button className="close-button" onClick={onClose} title="Close (Esc)">
            ✕
          </button>
        </div>

        {/* Search mode toggle */}
        <div className="search-mode-toggle">
          <button 
            className={`mode-button ${searchMode === 'text' ? 'active' : ''}`}
            onClick={() => setSearchMode('text')}
          >
            Text Search
          </button>
          <button 
            className={`mode-button ${searchMode === 'vector' ? 'active' : ''}`}
            onClick={() => setSearchMode('vector')}
          >
            Vector Search
          </button>
        </div>

        {/* Search input */}
        <div className="search-input-section">
          {searchMode === 'text' ? (
            <input 
              type="text"
              className="search-input"
              placeholder="Search motion files... (e.g., 'walking', 'dance', 'running')"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              autoFocus
            />
          ) : (
            <div className="vector-input-section">
              <textarea 
                className="vector-input"
                placeholder="Enter vector as JSON array: [0.1, 0.2, 0.3, 0.4, ...]"
                value={vectorInput}
                onChange={(e) => setVectorInput(e.target.value)}
                rows={3}
              />
              <button 
                className="vector-search-button"
                onClick={handleVectorSearch}
                disabled={!vectorInput.trim()}
              >
                Search
              </button>
            </div>
          )}
        </div>

        {/* Loading indicator */}
        {isSearching && (
          <div className="search-loading">
            <div className="spinner"></div>
            <span>Searching motion database...</span>
          </div>
        )}

        {/* Search results */}
        <div className="search-results">
          {searchResults.length > 0 ? (
            <>
              <div className="results-header">
                <span>Found {searchResults.length} motion(s)</span>
                <span className="drag-hint">💡 Drag files to BlendMixer</span>
              </div>
              
              <div className="results-list">
                {searchResults.map(result => (
                  <DraggableResult 
                    key={result.id}
                    result={result}
                    onSelect={handleFileSelect}
                  />
                ))}
              </div>
            </>
          ) : (
            !isSearching && searchQuery && (
              <div className="no-results">
                <p>No motion files found for "{searchQuery}"</p>
                <p>Try different search terms or switch to vector search</p>
              </div>
            )
          )}
        </div>

        {/* Instructions */}
        <div className="search-instructions">
          <h4>Search Tips:</h4>
          <ul>
            <li><strong>Text Search:</strong> Use keywords like 'walk', 'run', 'dance'</li>
            <li><strong>Vector Search:</strong> Input motion embedding vectors for similarity matching</li>
            <li><strong>Drag & Drop:</strong> Drag results directly to BlendMixer sequence slots</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
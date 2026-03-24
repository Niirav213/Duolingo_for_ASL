import React from 'react';
import './PathView.css';
import Mascot from '../components/Mascot';

// A structured path of learning specific ASL letters.
const ALPHABET_PATH = [
  { id: 'A', type: 'lesson', label: 'A' },
  { id: 'B', type: 'lesson', label: 'B' },
  { id: 'C', type: 'lesson', label: 'C' },
  { id: 'Test-1', type: 'test', label: 'Test 1', letters: ['A', 'B', 'C'] },
  { id: 'D', type: 'lesson', label: 'D' },
  { id: 'E', type: 'lesson', label: 'E' },
  { id: 'F', type: 'lesson', label: 'F' },
  { id: 'Test-2', type: 'test', label: 'Test 2', letters: ['D', 'E', 'F'] },
];

const PathView = ({ completedLevels, onStartLesson }) => {
  // Determine the status of each node
  const getLevelStatus = (levelId, index) => {
    if (completedLevels.includes(levelId)) return 'completed';
    // It's active if it's the very first node, OR if the previous node is completed
    if (index === 0) return 'active';
    const prevId = ALPHABET_PATH[index - 1].id;
    if (completedLevels.includes(prevId)) return 'active';
    return 'locked';
  };

  return (
    <div className="path-view">
      <div className="unit-container">
        <header className="unit-header" style={{ backgroundColor: 'var(--color-primary)' }}>
          <div>
            <h3>Unit 1</h3>
            <p>Learn the ASL Alphabet</p>
          </div>
          <button className="unit-guide-btn">Guidebook</button>
        </header>

        <div className="path-nodes">
          {ALPHABET_PATH.map((level, i) => {
            const status = getLevelStatus(level.id, i);
            const offset = Math.sin(i * 1.5) * 60;
            
            return (
              <div 
                key={level.id} 
                className={`path-node-wrapper ${status}`}
                style={{ transform: `translateX(${offset}px)` }}
              >
                {status === 'active' && (
                  <div className="mascot-indicator">
                    <Mascot variant="greeting" size="small" />
                  </div>
                )}
                
                <button 
                  className={`node-button ${status}`}
                  onClick={() => {
                    if (status === 'active' || status === 'completed') {
                      onStartLesson(level);
                    }
                  }}
                  style={{
                    backgroundColor: status === 'locked' ? 'var(--color-gray-light)' : (level.type === 'test' ? 'var(--color-secondary)' : 'var(--color-primary)')
                  }}
                >
                  <span className="node-icon">
                    {level.type === 'test' ? '📝' : level.label}
                  </span>
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default PathView;


import React from 'react';
import './Mascot.css';

const Mascot = ({ variant = 'default', size = 'medium' }) => {
  // In a real app, these would map to different PNGs or animated SVGs
  const getEmoji = () => {
    switch (variant) {
      case 'greeting': return '🦭👋';
      case 'happy': return '🦭✨';
      case 'sad': return '🦭💧';
      case 'celebrate': return '🦭🎉';
      default: return '🦭';
    }
  };

  return (
    <div className={`mascot mascot-${variant} mascot-${size}`}>
      {/* 
        Placeholder for the actual seal image specified by the user. 
        Will be replaced via <img src="/seal.png" alt="Signy the Seal" /> 
      */}
      <div className="mascot-placeholder">
        {getEmoji()}
      </div>
      {(variant === 'greeting' || variant === 'happy') && (
        <div className="mascot-sparkle">✨</div>
      )}
    </div>
  );
};

export default Mascot;

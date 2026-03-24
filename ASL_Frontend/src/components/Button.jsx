import React from 'react';
import './Button.css';

const Button = ({ children, variant = 'primary', size = 'medium', className = '', onClick, disabled }) => {
  return (
    <button
      className={`btn btn-${variant} btn-${size} ${className}`}
      onClick={onClick}
      disabled={disabled}
    >
      <span className="btn-content">{children}</span>
    </button>
  );
};

export default Button;

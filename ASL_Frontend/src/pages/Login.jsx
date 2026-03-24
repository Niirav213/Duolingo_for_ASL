import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store';
import './Login.css';

export const Login = ({ isRegister = false }) => {
  const navigate = useNavigate();
  const { login, register, isLoading, error } = useAuthStore();
  const [isRegisterMode, setIsRegisterMode] = useState(isRegister);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [localError, setLocalError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLocalError('');

    if (isRegisterMode) {
      if (!username || !email || !password || !confirmPassword) {
        setLocalError('All fields are required');
        return;
      }
      if (password !== confirmPassword) {
        setLocalError('Passwords do not match');
        return;
      }
      if (password.length < 6) {
        setLocalError('Password must be at least 6 characters');
        return;
      }

      try {
        await register(username, email, password);
        navigate('/');
      } catch (err) {
        setLocalError(err.message || 'Registration failed');
      }
    } else {
      if (!username || !password) {
        setLocalError('Username and password are required');
        return;
      }

      try {
        await login(username, password);
        navigate('/');
      } catch (err) {
        setLocalError(err.message || 'Login failed');
      }
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <h1>🤟 ASL Platform</h1>
          <p>Learn American Sign Language</p>
        </div>

        <h2>{isRegisterMode ? 'Create Account' : 'Login'}</h2>

        {(error || localError) && (
          <div className="error-box">
            <p>{error || localError}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
            />
          </div>

          {isRegisterMode && (
            <div className="form-group">
              <label>Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email"
              />
            </div>
          )}

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
            />
          </div>

          {isRegisterMode && (
            <div className="form-group">
              <label>Confirm Password</label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Confirm your password"
              />
            </div>
          )}

          <button type="submit" disabled={isLoading} className="submit-btn border-b-4">
            {isLoading ? 'Loading...' : isRegisterMode ? 'Create Account' : 'Login'}
          </button>
        </form>

        <div className="toggle-mode">
          <p>
            {isRegisterMode ? 'Already have an account? ' : "Don't have an account? "}
            <button
              type="button"
              onClick={() => {
                setIsRegisterMode(!isRegisterMode);
                setLocalError('');
              }}
              className="toggle-btn"
            >
              {isRegisterMode ? 'Login here' : 'Sign up here'}
            </button>
          </p>
        </div>

        <div className="demo-credentials">
          <p className="demo-title">Demo Credentials:</p>
          <p>Username: <span>demo</span></p>
          <p>Password: <span>demo123</span></p>
        </div>
      </div>
    </div>
  );
};

export default Login;

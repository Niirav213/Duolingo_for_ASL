import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store';

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
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white flex items-center justify-center p-6">
      <div className="w-full max-w-md">
        {/* Logo/Title */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">🤟 ASL Platform</h1>
          <p className="text-gray-400">Learn American Sign Language</p>
        </div>

        {/* Card */}
        <div className="bg-gray-800 bg-opacity-50 border border-gray-700 rounded-lg p-8 shadow-xl">
          <h2 className="text-2xl font-bold mb-6 text-center">
            {isRegisterMode ? 'Create Account' : 'Login'}
          </h2>

          {/* Error Message */}
          {(error || localError) && (
            <div className="bg-red-600 bg-opacity-20 border border-red-500 rounded-lg p-4 mb-6">
              <p className="text-red-400 text-sm">{error || localError}</p>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Username */}
            <div>
              <label className="block text-sm font-medium mb-2">Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500 focus:ring-opacity-50 text-white"
                placeholder="Enter your username"
              />
            </div>

            {/* Email (Register Only) */}
            {isRegisterMode && (
              <div>
                <label className="block text-sm font-medium mb-2">Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500 focus:ring-opacity-50 text-white"
                  placeholder="Enter your email"
                />
              </div>
            )}

            {/* Password */}
            <div>
              <label className="block text-sm font-medium mb-2">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500 focus:ring-opacity-50 text-white"
                placeholder="Enter your password"
              />
            </div>

            {/* Confirm Password (Register Only) */}
            {isRegisterMode && (
              <div>
                <label className="block text-sm font-medium mb-2">Confirm Password</label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500 focus:ring-opacity-50 text-white"
                  placeholder="Confirm your password"
                />
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded-lg transition-all duration-200 mt-6"
            >
              {isLoading ? 'Loading...' : isRegisterMode ? 'Create Account' : 'Login'}
            </button>
          </form>

          {/* Toggle Mode */}
          <div className="mt-6 text-center">
            <p className="text-gray-400 text-sm">
              {isRegisterMode ? 'Already have an account? ' : "Don't have an account? "}
              <button
                type="button"
                onClick={() => {
                  setIsRegisterMode(!isRegisterMode);
                  setLocalError('');
                }}
                className="text-purple-400 hover:text-purple-300 font-medium transition-colors"
              >
                {isRegisterMode ? 'Login here' : 'Sign up here'}
              </button>
            </p>
          </div>
        </div>

        {/* Demo Credentials */}
        <div className="mt-8 p-4 bg-blue-600 bg-opacity-10 border border-blue-500 rounded-lg">
          <p className="text-xs text-gray-300 mb-2 font-semibold">Demo Credentials:</p>
          <p className="text-xs text-gray-400">Username: <span className="text-blue-400">demo</span></p>
          <p className="text-xs text-gray-400">Password: <span className="text-blue-400">demo123</span></p>
        </div>
      </div>
    </div>
  );
};

export default Login;

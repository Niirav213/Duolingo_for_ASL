import React from 'react';
import Sidebar from './Sidebar';
import TopBar from './TopBar';

const Layout = ({ children, streak = 0, gems = 0, hearts = 5 }) => {
  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <TopBar streak={streak} gems={gems} hearts={hearts} />
        <main className="content-area">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;

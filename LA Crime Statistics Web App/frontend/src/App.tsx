import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Home from './components/Home';
import SafetyScore from './components/SafetyScore';
import FilteredData from './components/FilteredData';
import Discussion from './components/Discussion';
import About from './components/About';

const App: React.FC = () => {
  return (
    <Router>
      <div style={{ backgroundColor: '#f9f9f9' }}> {/* Light Grey Background */}
        {/* Header */}
        <header style={{ textAlign: 'center', padding: '1px', backgroundColor: '#111', color: '#fff', letterSpacing: '0px'}}>
        <h1 style={{ fontSize: '2.25em'}}>Los Angeles Crime Statistics</h1>
        </header>

        {/* Navigation Banner */}
        <nav style={{ backgroundColor: '#222', color: '#fff', padding: '8.5px', borderBottom: '2px solid #555' }}>
          <ul style={{ listStyleType: 'none', display: 'flex', justifyContent: 'center', margin: '0', padding: '0' }}>
            <li style={{ margin: '0 50px' }}>
              <Link to="/" style={{ textDecoration: 'none', color: 'white', letterSpacing: '0.5px' }}>Home</Link>
            </li>
            <li style={{ margin: '0 50px' }}>
              <Link to="/safety-score" style={{ textDecoration: 'none', color: 'white', letterSpacing: '0.5px' }}>Safety Score</Link>
            </li>
            <li style={{ margin: '0 50px' }}>
              <Link to="/filtered-data" style={{ textDecoration: 'none', color: 'white', letterSpacing: '0.5px' }}>Filtered Data</Link>
            </li>
            <li style={{ margin: '0 50px' }}>
              <Link to="/discussion" style={{ textDecoration: 'none', color: 'white', letterSpacing: '0.5px' }}>Discussion</Link>
            </li>
            <li style={{ margin: '0 50px' }}>
              <Link to="/about" style={{ textDecoration: 'none', color: 'white', letterSpacing: '0.5px' }}>About</Link>
            </li>
          </ul>
        </nav>

        {/* Routes */}
        <Routes>
          <Route path="/" Component={Home} />
          <Route path="/safety-score" Component={SafetyScore} />
          <Route path="/filtered-data" Component={FilteredData} />
          <Route path="/discussion" Component={Discussion} />
          <Route path="/about" Component={About} />
        </Routes>

        <style>
          {`
            nav ul li a:hover {
              background-color: #333;
            }
          `}
        </style>
      </div>
    </Router>
  );
};

export default App;


// src/components/SafetyScore.tsx

import React, { useState } from 'react';

const SafetyScore: React.FC = () => {
  const [zipCode, setZipCode] = useState('');
  const [safetyScore, setSafetyScore] = useState<number | null>(null);
  const [error, setError] = useState('');

  const isZipCodeInRange = (zip: string): boolean => {
    const zipRanges = [
      { start: 90001, end: 90099 },
      { start: 90189, end: 90813 },
      { start: 91040, end: 91609 },
    ];
    const numericZip = parseInt(zip, 10);

    for (const range of zipRanges) {
      if (numericZip >= range.start && numericZip <= range.end) {
        return true;
      }
    }
    return false;
  };

  const handleFormSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!/^\d{5}$/.test(zipCode)) {
      setError('Please enter a valid 5-digit ZIP code consisting of numbers.');
      return;
    }

    if (!isZipCodeInRange(zipCode)) {
      setError('Please enter a ZIP code within the allowed ranges.');
      return;
    }

    try {
      const response = await fetch(`/api/safety-score/${zipCode}`);
      const data = await response.json();
      // Assuming the response includes a 'safetyScore' field
      setSafetyScore(data.safetyScore);
      setError('');
    } catch (error) {
      console.error('Error fetching safety score:', error);
      setError('Error fetching safety score. Please try again.');
    }
  };

  const sectionStyles: React.CSSProperties = {
    width: '100%',
    maxWidth: '800px',
    margin: '0 auto',
    textAlign: 'left',
    padding: '10px',
    marginBottom: '20px',
    backgroundColor: '#f2f2f2',
    borderRadius: '10px',
    color: 'black',
    lineHeight: '1.75',
  };
  const sectionHeaderStyles: React.CSSProperties = {
    fontSize: '1.5em',
    marginBottom: '10px',
    marginTop: '5px',
    color: 'black',
  };
  const submitButtonContainerStyles: React.CSSProperties = {
    marginTop: '20px', // Add space after the submit button
  };

  return (
    <div style={{ minHeight: 'calc(100vh - 100px)', padding: '10px', textAlign: 'center' }}>
      <h1 style={{ fontSize: '2em', margin: '20px 0' }}>Safety Score Calculator</h1>

      <form onSubmit={handleFormSubmit} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <label style={{ marginBottom: '10px' }}>
          Enter ZIP Code:&nbsp;&nbsp;&nbsp;
          <input
            type="text"
            value={zipCode}
            onChange={(e) => setZipCode(e.target.value)}
            maxLength={5}
            style={{ marginRight: '10px' }}
          />
        </label>
        <button type="submit">Calculate Safety Score</button>
        {error && <p style={{ color: 'red', marginTop: '10px' }}>{error}</p>}
      </form>

      <div style={submitButtonContainerStyles}>
        <div style={{ ...sectionStyles }}>
          <h2 style={sectionHeaderStyles}>Instructions</h2>
          <p>
            Please enter a 5-digit Los Angeles area ZIP code to calculate the <b>safety score</b> for that area.
          </p>
        </div>
      </div>

      <div style={{ marginTop: '20px', ...sectionStyles }}>
        <h2 style={sectionHeaderStyles}>Purpose</h2>
        <p>
          The Safety Score Calculator is intended for community members of Los Angeles to gain insights into the safety levels of their neighborhoods, aiding in understanding and fostering a safer community environment.
        </p>
      </div>

      {safetyScore !== null && (
        <div style={{ marginTop: '20px', ...sectionStyles }}>
          <h2 style={sectionHeaderStyles}>Safety Score</h2>
          <p>{safetyScore}</p>
        </div>
      )}
    </div>
  );
};

export default SafetyScore;
// src/components/FilteredData.tsx

import React, { useState, useEffect } from 'react';

const FilteredData: React.FC = () => {
  const [selectedFilter, setSelectedFilter] = useState<string>('');
  const [crimeData, setCrimeData] = useState<any[]>([]);
  const [error, setError] = useState<string>('');

  const fetchCrimeData = async (selectedOption: string) => {
    try {
      // Simulating error since the API fetching logic is not implemented
      setError('Error: API fetching not implemented. Please select a filter.');
      setCrimeData([]); // Clear previous data
    } catch (error) {
      console.error('Error fetching data:', error);
      setError('Error fetching data. Please try again.');
    }
  };

  useEffect(() => {
    if (selectedFilter) {
      fetchCrimeData(selectedFilter);
    }
  }, [selectedFilter]);

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

  const spaceAfterDropdownStyles: React.CSSProperties = {
    marginTop: '20px',
  };

  return (
    <div style={{ minHeight: 'calc(100vh - 100px)', padding: '10px', textAlign: 'center' }}>
      <h1 style={{ fontSize: '2em', margin: '20px 0' }}>Filtered Data Page</h1>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <select
          value={selectedFilter}
          onChange={(e) => setSelectedFilter(e.target.value)}
          style={{ width: '350px', padding: '2px', fontSize: '0.9em' }}
        >
          <option value="">Select Filter</option>
          <option value="highestCrimeAreas">Areas with Highest Crime Rates</option>
          <option value="highestTimeOfDay">Time of Day when Crime is Highest</option>
          <option value="mostPrevalentCrime">Most Prevalent Crimes Committed</option>
          <option value="highestGrandTheftAreas">Areas where Grand Theft Auto is Highest</option>
          <option value="highestBatteryAreas">Areas where Battery is Highest</option>
          <option value="mostPrevalentAge">Most Prevalent Victim Age</option>
          {/* Add other options */}
        </select>
        {error && <p style={{ color: 'red', marginTop: '10px' }}>{error}</p>}
      </div>

      <div style={{ ...spaceAfterDropdownStyles }}>
        <div style={{ height: '25px' }}></div>
      </div>

      <div style={{ marginTop: '20px', ...sectionStyles }}>
        <h2 style={sectionHeaderStyles}>Instructions</h2>
        <p>
          Please select a filter from the dropdown menu to view specific crime statistics in the selected category.
        </p>
      </div>

      <div style={{ marginTop: '20px', ...sectionStyles }}>
        <h2 style={sectionHeaderStyles}>Purpose</h2>
        <p>
          This tool aims to provide insights into different crime statistics based on selected filters,
          assisting users in understanding various aspects of crime data within the specified areas.
        </p>
      </div>

      {/* Display fetched data in a table */}
      <table>
        {/* Table content based on fetched data */}
      </table>
    </div>
  );
};

export default FilteredData;
// src/components/Home.tsx

import React from 'react';
import losAngelesImage from '../images/Getty_515070156_EDITORIALONLY_LosAngeles_HollywoodBlvd_Web72DPI_0.jpg';

const Home: React.FC = () => {
  return (
    <div style={{ minHeight: 'calc(100vh - 100px)', padding: '35px', textAlign: 'center' }}>
      <img src={losAngelesImage} alt="Los Angeles" style={{ maxWidth: '700px', height: 'auto' }} />
      <div style={{ maxWidth: '800px', margin: '0 auto', textAlign: 'left', lineHeight: '1.75' }}>
      <div style={{ textAlign: 'center' }}>
          <h2>Welcome Los Angeles community members!</h2>
        </div>
      <p style={{ textAlign: 'center'}}>
        Come explore and understand the safety dynamics of our city with precision! This platform serves as a vital tool for every member of our community to access comprehensive crime statistics within Los Angeles.
      </p>
      <p style={{ textAlign: 'center'}}>
      The <b>Safety Score</b> feature empowers you to gauge the safety of your area by simply entering your address. Get instant insights into the safety level, ensuring you're informed and aware.
      </p>
      <p style={{ textAlign: 'center'}}>
      Delve deeper into the city's crime landscape through the <b>Filtered Data</b> page. Uncover intriguing trends and unexpected crime statistics, allowing you to better understand the dynamics shaping our community. 
      </p>
      <p style={{ textAlign: 'center'}}>
      Engage in meaningful conversations on the <b>Discussion</b> page. Share observations, report incidents, and connect with fellow residents to collectively contribute to a safer environment.
      </p>
      <p style={{ textAlign: 'center'}}>
      Curious about our mission and functionalities? Navigate to the <b>About</b> page for a detailed overview of our website's capabilities and how it serves the community.
      </p>
      <p style={{ textAlign: 'center'}}>
      Join us in our mission to cultivate a safer Los Angeles. Together, armed with knowledge, discussions, and insights, let's build stronger and safer communities for a brighter future.
      </p>
    </div>
  </div>
  );
};

export default Home;
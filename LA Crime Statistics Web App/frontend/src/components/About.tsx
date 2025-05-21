import React from 'react';

const About: React.FC = () => {
  const sectionStyles: React.CSSProperties = {
    width: '100%',
    maxWidth: '800px',
    margin: '0 auto',
    textAlign: 'left',
    padding: '10px',
    marginBottom: '20px',
    backgroundColor: '#f2f2f2', // Background color
    borderRadius: '10px', // Border radius
    color: 'black',
    lineHeight:'1.75'
  };
  const sectionHeaderStyles: React.CSSProperties = {
    fontSize: '1.5em',
    marginBottom: '10px',
    marginTop: '5px', // Adjust the margin-top to reduce space above section headers
    color: 'black',
  };
  const contentPaddingStyles: React.CSSProperties = {
    paddingTop: '10px', // Adjust the padding-top to increase space after the "About Us" title
  };

  return (
    <div style={{ minHeight: 'calc(100vh - 100px)', padding: '10px', textAlign: 'center' }}>
      <h1 style={{ fontSize: '2em', margin: '20px 0' }}>About Us</h1>
      <div style={contentPaddingStyles}>

      {/* Purpose and Functionality Section */}
      <section style={sectionStyles}>
        <h2 style={sectionHeaderStyles}>Purpose and Functionality</h2>
        <p>
        Our platform "Los Angeles Crime Statistics" is a comprehensive resource developed to serve the community within Los Angeles. Our primary goal is to provide residents with crucial crime statistics, fostering awareness and safety in their neighborhoods. By offering insight into local crime trends, we empower individuals to make informed decisions regarding their security and contribute to a safer community.
        </p>
      </section>

      {/* Features and Utilities Section */}
      <section style={sectionStyles}>
        <h2 style={sectionHeaderStyles}>Features and Utilities</h2>
        <p>
        The platform provides extensive crime statistics, allowing users to understand and analyze crime occurrences across various areas in Los Angeles. This data plays a pivotal role in enhancing awareness and vigilance, working toward creating a secure environment for all residents. Additionally, our discussion panel encourages active community participation, facilitating conversations about observations, incident reporting, and collective efforts to ensure neighborhood safety.
        </p>
      </section>

      {/* Importance of Staying Updated Section */}
      <section style={sectionStyles}>
        <h2 style={sectionHeaderStyles}>Awareness</h2>
        <p>
        Recent statistics highlight an alarming trend in the increased crime rate within specific districts of Los Angeles. Our platform emphasizes the criticality of staying abreast of evolving crime patterns. This information enables proactive measures and readiness to address potential safety concerns.
        </p>
      </section>

      {/* Technology Stack Section */}
      <section style={sectionStyles}>
        <h2 style={sectionHeaderStyles}>Technology Stack</h2>
        <p>
        Developed using modern technologies like React, TypeScript, Node.js, and SQL, Los Angeles Crime Statistics seamlessly integrates each component to build a functional, user-friendly interface. The platform is a collaborative effort, shaped by the expertise and dedication of Abil, Arnold, Nathan, and Santhra.
        </p>
      </section>

      {/* Conclusion Section */}
      <section style={sectionStyles}>
        <h2 style={sectionHeaderStyles}>Conclusion</h2>
        <p>
        Los Angeles Crime Statistics is our commitment to community welfare. It stands as a testament to our belief in the power of information and collaboration to build safer neighborhoods. Join us in our mission to cultivate a safer Los Angeles!
        </p>
      </section>
    </div>
    </div>
  );
};

export default About;
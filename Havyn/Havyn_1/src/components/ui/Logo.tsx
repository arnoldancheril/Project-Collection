import React from 'react';
import { Image, StyleSheet, View, ImageSourcePropType, StyleProp, ViewStyle, ImageStyle } from 'react-native';

type LogoType = 'icon' | 'horizontal';

interface LogoProps {
  type?: LogoType;
  size?: number;
  style?: StyleProp<ViewStyle>;
  imageStyle?: StyleProp<ImageStyle>;
}

const Logo: React.FC<LogoProps> = ({ 
  type = 'icon', 
  size = 120,
  style,
  imageStyle
}) => {
  const logoSource: ImageSourcePropType = 
    type === 'icon' 
      ? require('../../../assets/images/Logo_Just_Icon.png')
      : require('../../../assets/images/Logo_Icon_With_Name_on_Left.png');
  
  return (
    <View style={[styles.container, style]}>
      <Image 
        source={logoSource} 
        style={[
          styles.image, 
          { width: type === 'icon' ? size : size * 2, height: size },
          imageStyle
        ]}
        resizeMode={type === 'icon' ? 'contain' : 'contain'}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  image: {
    width: 120,
    height: 120,
  },
});

export default Logo; 
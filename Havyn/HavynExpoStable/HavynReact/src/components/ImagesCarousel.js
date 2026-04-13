import React, { useState, useRef } from 'react';
import {
  View,
  Image,
  StyleSheet,
  Dimensions,
  FlatList,
  TouchableOpacity,
  Text
} from 'react-native';
import { COLORS } from '../utils/theme';

const { width } = Dimensions.get('window');

const ImagesCarousel = ({ images }) => {
  const [activeIndex, setActiveIndex] = useState(0);
  const flatListRef = useRef(null);

  // Ensure we have at least one image
  const displayImages = images && images.length > 0 
    ? images 
    : [require('../../assets/person-placeholder.jpg')];

  const onViewableItemsChanged = useRef(({ viewableItems }) => {
    if (viewableItems && viewableItems.length > 0) {
      setActiveIndex(viewableItems[0].index);
    }
  }).current;

  const viewabilityConfig = useRef({
    itemVisiblePercentThreshold: 50
  }).current;

  const scrollToIndex = (index) => {
    if (flatListRef.current) {
      flatListRef.current.scrollToIndex({
        index,
        animated: true
      });
    }
  };

  return (
    <View style={styles.container}>
      <FlatList
        ref={flatListRef}
        data={displayImages}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        keyExtractor={(_, index) => index.toString()}
        renderItem={({ item }) => (
          <View style={styles.imageContainer}>
            <Image
              source={item}
              style={styles.image}
              resizeMode="cover"
            />
          </View>
        )}
        onViewableItemsChanged={onViewableItemsChanged}
        viewabilityConfig={viewabilityConfig}
      />
      
      {/* Image indicator dots */}
      {displayImages.length > 1 && (
        <View style={styles.pagination}>
          {displayImages.map((_, index) => (
            <TouchableOpacity 
              key={index}
              style={[
                styles.paginationDot,
                activeIndex === index && styles.paginationDotActive
              ]}
              onPress={() => scrollToIndex(index)}
            />
          ))}
        </View>
      )}
      
      {/* Image counter */}
      {displayImages.length > 1 && (
        <View style={styles.counter}>
          <Text style={styles.counterText}>
            {activeIndex + 1} / {displayImages.length}
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    height: '100%',
    position: 'relative',
  },
  imageContainer: {
    width,
    height: '100%',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  pagination: {
    position: 'absolute',
    bottom: 15,
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'center',
  },
  paginationDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    marginHorizontal: 4,
  },
  paginationDotActive: {
    backgroundColor: COLORS.primary,
    width: 12,
    height: 8,
  },
  counter: {
    position: 'absolute',
    top: 15,
    left: 15,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
  },
  counterText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
  }
});

export default ImagesCarousel; 
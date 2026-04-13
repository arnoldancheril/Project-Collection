import React from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES, SHADOWS } from '../utils/theme';

const ChatBubble = ({
  message,
  isUser,
  showAvatar = true,
  timestamp,
  status,
  onImagePress,
}) => {
  const formatTime = (date) => {
    if (!date) return '';
    const messageDate = typeof date === 'object' ? date : new Date(date);
    
    return messageDate.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    });
  };
  
  const renderStatus = () => {
    if (isUser) {
      if (status === 'sent') {
        return <Ionicons name="checkmark-outline" size={12} color={COLORS.textSecondary} />;
      } else if (status === 'delivered') {
        return <Ionicons name="checkmark-done-outline" size={12} color={COLORS.textSecondary} />;
      } else if (status === 'read') {
        return <Ionicons name="checkmark-done-outline" size={12} color={COLORS.primary} />;
      }
    }
    return null;
  };
  
  const isImage = message.type === 'image' && message.imageUrl;
  const isLocation = message.type === 'location' && message.locationData;
  
  return (
    <View style={[
      styles.container,
      isUser ? styles.userContainer : styles.otherContainer
    ]}>
      {!isUser && showAvatar && (
        <Image 
          source={{ 
            uri: message.senderPhotoUrl || 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1740&q=80' 
          }}
          style={styles.avatar}
        />
      )}
      
      <View style={[
        styles.bubbleContainer,
        isUser ? styles.userBubble : styles.otherBubble,
        isImage && styles.imageBubble,
        isLocation && styles.locationBubble,
      ]}>
        {isImage ? (
          <TouchableOpacity onPress={onImagePress} activeOpacity={0.9}>
            <Image
              source={{ uri: message.imageUrl }}
              style={styles.messageImage}
              resizeMode="cover"
            />
          </TouchableOpacity>
        ) : isLocation ? (
          <TouchableOpacity onPress={() => {}} style={styles.locationContainer}>
            <View style={styles.locationPreview}>
              <Ionicons name="location" size={24} color={COLORS.primary} />
            </View>
            <Text style={styles.locationText}>Location shared</Text>
            <Text style={styles.locationAddress} numberOfLines={2}>
              {message.locationData.name || 'View on map'}
            </Text>
          </TouchableOpacity>
        ) : (
          <Text style={[
            styles.messageText,
            isUser ? styles.userMessageText : styles.otherMessageText
          ]}>
            {message.text}
          </Text>
        )}
        
        <View style={styles.timeContainer}>
          <Text style={styles.timeText}>{formatTime(timestamp)}</Text>
          {renderStatus()}
        </View>
      </View>
      
      {isUser && showAvatar && (
        <View style={styles.avatarPlaceholder} />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    marginVertical: SIZES.base / 2,
    alignItems: 'flex-end',
  },
  userContainer: {
    justifyContent: 'flex-end',
    marginLeft: 50,
  },
  otherContainer: {
    justifyContent: 'flex-start',
    marginRight: 50,
  },
  avatar: {
    width: 28,
    height: 28,
    borderRadius: 14,
    marginRight: SIZES.base / 2,
  },
  avatarPlaceholder: {
    width: 28,
    height: 28,
    marginLeft: SIZES.base / 2,
  },
  bubbleContainer: {
    borderRadius: SIZES.radius,
    padding: SIZES.base,
    maxWidth: '80%',
    ...SHADOWS.small,
  },
  userBubble: {
    backgroundColor: COLORS.primary,
    borderBottomRightRadius: 0,
  },
  otherBubble: {
    backgroundColor: COLORS.surface,
    borderBottomLeftRadius: 0,
  },
  imageBubble: {
    padding: SIZES.base / 2,
    backgroundColor: 'transparent',
    overflow: 'hidden',
  },
  locationBubble: {
    padding: SIZES.base,
    width: 200,
  },
  messageText: {
    fontSize: SIZES.font,
  },
  userMessageText: {
    color: COLORS.surface,
  },
  otherMessageText: {
    color: COLORS.text,
  },
  messageImage: {
    width: 200,
    height: 200,
    borderRadius: SIZES.radius,
  },
  locationContainer: {
    width: '100%',
  },
  locationPreview: {
    height: 100,
    backgroundColor: COLORS.background,
    borderRadius: SIZES.radius,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: SIZES.base / 2,
  },
  locationText: {
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: SIZES.base / 4,
  },
  locationAddress: {
    fontSize: SIZES.small,
    color: COLORS.textSecondary,
  },
  timeContainer: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    alignItems: 'center',
    marginTop: SIZES.base / 2,
  },
  timeText: {
    fontSize: 10,
    color: isUser => isUser ? 'rgba(255, 255, 255, 0.7)' : COLORS.textSecondary,
    marginRight: 2,
  },
});

export default ChatBubble; 
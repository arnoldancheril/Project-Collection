import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Image,
  SafeAreaView
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../../utils/theme';

// Sample messages for chat screen
const generateSampleMessages = (chatPartner) => {
  // Safely get the first name or use a default
  const firstName = chatPartner?.name?.split(' ')[0] || 'User';
  const isElsa = firstName === 'Elsa';
  
  const messages = [
    {
      id: '1',
      text: `Hi! I saw your listing and I'm interested in the ${isElsa ? 'River North' : 'apartment'}. Is it still available?`,
      timestamp: '10:30 AM',
      isUser: true,
    },
    {
      id: '2',
      text: `Hello! Yes, the place is still available. Would you like to know more about it?`,
      timestamp: '10:32 AM',
      isUser: false,
    },
    {
      id: '3',
      text: 'Great! I have a few questions. What are the move-in costs? And are utilities included in the rent?',
      timestamp: '10:35 AM',
      isUser: true,
    },
    {
      id: '4',
      text: `The move-in costs include first month's rent and a security deposit equal to one month's rent. Water and trash are included, but electricity and internet are separate.`,
      timestamp: '10:38 AM',
      isUser: false,
    },
    {
      id: '5',
      text: 'That sounds reasonable. Is the apartment furnished or unfurnished?',
      timestamp: '10:40 AM',
      isUser: true,
    },
    {
      id: '6',
      text: 'The apartment comes partially furnished with a bed, dresser, and desk in the bedroom. The living room has a couch and coffee table.',
      timestamp: '10:42 AM',
      isUser: false,
    },
    {
      id: '7',
      text: 'Perfect! When would it be possible to schedule a viewing?',
      timestamp: '10:45 AM',
      isUser: true,
    },
  ];
  
  return messages;
};

const ChatScreen = ({ route, navigation }) => {
  // Safely access route parameters with default values
  const chat = route.params?.profile || route.params?.chat || {
    id: '0',
    name: 'Chat Partner',
    avatar: require('../../../assets/person-placeholder.jpg')
  };
  
  const [messages, setMessages] = useState(generateSampleMessages(chat));
  const [inputText, setInputText] = useState('');
  const flatListRef = useRef(null);

  const sendMessage = () => {
    if (inputText.trim() === '') return;
    
    const newMessage = {
      id: Date.now().toString(),
      text: inputText.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isUser: true,
    };
    
    setMessages([...messages, newMessage]);
    setInputText('');
    
    // Simulate receiving a response
    setTimeout(() => {
      const responseMessage = {
        id: (Date.now() + 1).toString(),
        text: "That sounds great! I'm available for a viewing tomorrow afternoon or Friday morning. Which works better for you?",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isUser: false,
      };
      
      setMessages(prevMessages => [...prevMessages, responseMessage]);
    }, 1500);
  };

  const renderMessage = ({ item }) => (
    <View style={[
      styles.messageBubble,
      item.isUser ? styles.userMessage : styles.partnerMessage
    ]}>
      <Text style={[
        styles.messageText, 
        item.isUser ? { color: 'white' } : { color: COLORS.text }
      ]}>
        {item.text}
      </Text>
      <Text style={[
        styles.messageTimestamp,
        item.isUser ? { color: 'rgba(255,255,255,0.7)' } : { color: '#999' }
      ]}>
        {item.timestamp}
      </Text>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      {/* Chat Header */}
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="chevron-back" size={28} color={COLORS.text} />
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.headerProfile} onPress={() => {}}>
          <Image 
            source={chat.avatar || require('../../../assets/person-placeholder.jpg')} 
            style={styles.profileImage} 
          />
          <View style={styles.profileInfo}>
            <Text style={styles.profileName}>{chat.name || 'Chat Partner'}</Text>
            <Text style={styles.profileStatus}>Online</Text>
          </View>
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.optionsButton} onPress={() => {}}>
          <Ionicons name="ellipsis-vertical" size={24} color={COLORS.text} />
        </TouchableOpacity>
      </View>
      
      {/* Messages */}
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.messageList}
        onContentSizeChange={() => flatListRef.current.scrollToEnd({ animated: true })}
        onLayout={() => flatListRef.current.scrollToEnd({ animated: true })}
      />
      
      {/* Input Area */}
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        <View style={styles.inputContainer}>
          <TouchableOpacity style={styles.attachButton}>
            <Ionicons name="add-circle-outline" size={24} color={COLORS.primary} />
          </TouchableOpacity>
          
          <TextInput
            style={styles.textInput}
            placeholder="Type a message..."
            value={inputText}
            onChangeText={setInputText}
            multiline
            maxLength={500}
          />
          
          <TouchableOpacity 
            style={[
              styles.sendButton,
              inputText.trim() === '' ? styles.sendButtonDisabled : {}
            ]}
            onPress={sendMessage}
            disabled={inputText.trim() === ''}
          >
            <Ionicons name="send" size={20} color={inputText.trim() === '' ? '#999' : 'white'} />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eaeaea',
  },
  backButton: {
    padding: 5,
  },
  headerProfile: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 10,
  },
  profileImage: {
    width: 40,
    height: 40,
    borderRadius: 20,
  },
  profileInfo: {
    marginLeft: 10,
  },
  profileName: {
    fontWeight: '600',
    fontSize: 16,
    color: COLORS.text,
  },
  profileStatus: {
    fontSize: 12,
    color: '#4CAF50',
  },
  optionsButton: {
    padding: 5,
  },
  messageList: {
    padding: 15,
  },
  messageBubble: {
    maxWidth: '75%',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 18,
    marginBottom: 10,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: COLORS.primary,
    borderBottomRightRadius: 4,
  },
  partnerMessage: {
    alignSelf: 'flex-start',
    backgroundColor: 'white',
    borderBottomLeftRadius: 4,
  },
  messageText: {
    fontSize: 16,
    color: props => props.isUser ? 'white' : COLORS.text,
  },
  messageTimestamp: {
    fontSize: 11,
    color: props => props.isUser ? 'rgba(255,255,255,0.7)' : '#999',
    alignSelf: 'flex-end',
    marginTop: 5,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderTopWidth: 1,
    borderTopColor: '#eaeaea',
  },
  attachButton: {
    marginRight: 10,
  },
  textInput: {
    flex: 1,
    backgroundColor: '#f0f0f0',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 8,
    maxHeight: 100,
  },
  sendButton: {
    marginLeft: 10,
    backgroundColor: COLORS.primary,
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: '#e0e0e0',
  },
});

export default ChatScreen; 
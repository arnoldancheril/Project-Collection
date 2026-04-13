import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Switch,
  ScrollView,
  SafeAreaView,
  KeyboardAvoidingView,
  Platform
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../../utils/theme';

const FOCUS_OPTIONS = [
  {id: '1', name: 'Properties', description: 'Discuss different property listings'},
  {id: '2', name: 'Roommates', description: 'Find potential roommates for your place'},
  {id: '3', name: 'Neighborhoods', description: 'Information about different areas in the city'},
  {id: '4', name: 'Pricing', description: 'Discuss rental prices and affordability'},
  {id: '5', name: 'Amenities', description: 'Compare features of different properties'}
];

const CreateGroupChat = ({ navigation }) => {
  const [groupName, setGroupName] = useState('');
  const [groupDescription, setGroupDescription] = useState('');
  const [isPublic, setIsPublic] = useState(true);
  const [selectedFocus, setSelectedFocus] = useState(['1', '2']); // Default selections
  
  const toggleFocus = (id) => {
    if (selectedFocus.includes(id)) {
      setSelectedFocus(selectedFocus.filter(item => item !== id));
    } else {
      setSelectedFocus([...selectedFocus, id]);
    }
  };
  
  const createGroup = () => {
    // Create a new group chat object
    const newGroup = {
      id: Date.now().toString(),
      name: groupName,
      members: 1,
      lastMessage: "Group created just now",
      time: "Just now"
    };
    
    // Navigate back to the matches screen and pass the new group
    navigation.navigate('MatchesMain', { newGroup });
  };
  
  const canCreateGroup = groupName.trim().length > 0;
  
  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardAvoid}
      >
        <View style={styles.header}>
          <TouchableOpacity 
            style={styles.cancelButton}
            onPress={() => navigation.goBack()}
          >
            <Text style={styles.cancelText}>Cancel</Text>
          </TouchableOpacity>
          
          <Text style={styles.headerTitle}>Create Group Chat</Text>
          
          <TouchableOpacity 
            style={[styles.createButton, !canCreateGroup && styles.createButtonDisabled]}
            onPress={createGroup}
            disabled={!canCreateGroup}
          >
            <Text style={[styles.createText, !canCreateGroup && styles.createTextDisabled]}>
              Create
            </Text>
          </TouchableOpacity>
        </View>
        
        <ScrollView style={styles.form}>
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Group Details</Text>
            
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="Group Name"
                value={groupName}
                onChangeText={setGroupName}
                maxLength={50}
              />
              <Text style={styles.charCount}>{groupName.length}/50</Text>
            </View>
            
            <View style={styles.textAreaContainer}>
              <TextInput
                style={styles.textArea}
                placeholder="Group Description (optional)"
                value={groupDescription}
                onChangeText={setGroupDescription}
                multiline
                numberOfLines={4}
                maxLength={200}
              />
              <Text style={styles.charCount}>{groupDescription.length}/200</Text>
            </View>
          </View>
          
          <View style={styles.divider} />
          
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Privacy</Text>
            
            <View style={styles.toggleContainer}>
              <View style={styles.toggleTextContainer}>
                <Text style={styles.toggleLabel}>Public Group</Text>
                <Text style={styles.toggleDescription}>
                  {isPublic 
                    ? "Anyone can find and join this group" 
                    : "Only invited members can join"}
                </Text>
              </View>
              <Switch
                value={isPublic}
                onValueChange={setIsPublic}
                trackColor={{ false: '#d1d1d1', true: `${COLORS.primary}80` }}
                thumbColor={isPublic ? COLORS.primary : '#f4f3f4'}
                ios_backgroundColor="#d1d1d1"
              />
            </View>
          </View>
          
          <View style={styles.divider} />
          
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Focus</Text>
            <Text style={styles.sectionDescription}>
              This group will focus on discussing:
            </Text>
            
            {FOCUS_OPTIONS.map(option => (
              <TouchableOpacity 
                key={option.id}
                style={styles.checkItem}
                onPress={() => toggleFocus(option.id)}
              >
                <View style={styles.checkTextContainer}>
                  <Text style={styles.checkLabel}>{option.name}</Text>
                  <Text style={styles.checkDescription}>{option.description}</Text>
                </View>
                
                <View style={[
                  styles.checkbox, 
                  selectedFocus.includes(option.id) && styles.checkboxSelected
                ]}>
                  {selectedFocus.includes(option.id) && (
                    <Ionicons name="checkmark" size={16} color="#fff" />
                  )}
                </View>
              </TouchableOpacity>
            ))}
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  keyboardAvoid: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#eaeaea',
    backgroundColor: '#fff',
  },
  headerTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: COLORS.text,
  },
  cancelButton: {
    padding: 5,
  },
  cancelText: {
    fontSize: 16,
    color: COLORS.text,
  },
  createButton: {
    padding: 5,
  },
  createButtonDisabled: {
    opacity: 0.5,
  },
  createText: {
    fontSize: 16,
    color: COLORS.primary,
    fontWeight: '600',
  },
  createTextDisabled: {
    color: '#999',
  },
  form: {
    flex: 1,
  },
  section: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 16,
    color: COLORS.text,
  },
  sectionDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 12,
  },
  inputContainer: {
    position: 'relative',
    marginBottom: 16,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    backgroundColor: '#fafafa',
  },
  textAreaContainer: {
    position: 'relative',
  },
  textArea: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    backgroundColor: '#fafafa',
    minHeight: 100,
    textAlignVertical: 'top',
  },
  charCount: {
    position: 'absolute',
    bottom: 8,
    right: 12,
    fontSize: 12,
    color: '#999',
  },
  divider: {
    height: 8,
    backgroundColor: '#f5f5f5',
  },
  toggleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  toggleTextContainer: {
    flex: 1,
  },
  toggleLabel: {
    fontSize: 16,
    color: COLORS.text,
  },
  toggleDescription: {
    fontSize: 13,
    color: '#999',
    marginTop: 4,
  },
  checkItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  checkTextContainer: {
    flex: 1,
  },
  checkLabel: {
    fontSize: 16,
    color: COLORS.text,
  },
  checkDescription: {
    fontSize: 13,
    color: '#999',
    marginTop: 2,
  },
  checkbox: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#ddd',
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkboxSelected: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
});

export default CreateGroupChat; 
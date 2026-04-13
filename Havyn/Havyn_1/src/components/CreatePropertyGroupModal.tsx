import React, { useState } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  Modal, 
  TouchableOpacity, 
  TextInput,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Alert
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Listing } from '../models/Listing';

interface CreatePropertyGroupModalProps {
  listing: Listing | null;
  visible: boolean;
  onClose: () => void;
  onCreateGroup: (groupName: string, description: string, preferredRoommates: number) => void;
}

const CreatePropertyGroupModal: React.FC<CreatePropertyGroupModalProps> = ({ 
  listing, 
  visible, 
  onClose,
  onCreateGroup
}) => {
  const [groupName, setGroupName] = useState('');
  const [description, setDescription] = useState('');
  const [preferredRoommates, setPreferredRoommates] = useState('2');

  if (!listing) return null;

  const handleCreate = () => {
    // Validate inputs
    if (!groupName.trim()) {
      Alert.alert('Missing Information', 'Please enter a group name');
      return;
    }
    
    if (!description.trim()) {
      Alert.alert('Missing Information', 'Please enter a group description');
      return;
    }
    
    const roommates = parseInt(preferredRoommates);
    if (isNaN(roommates) || roommates < 1) {
      Alert.alert('Invalid Input', 'Please enter a valid number of roommates');
      return;
    }
    
    // Call the onCreateGroup function with the form data
    onCreateGroup(groupName, description, roommates);
    
    // Reset form fields
    setGroupName('');
    setDescription('');
    setPreferredRoommates('2');
  };

  // Generate a default group name based on the property
  const suggestedGroupName = `${listing.area} Housing Group`;

  // Generate a default description
  const suggestedDescription = `Looking for roommates to share a ${listing.homeDetails.rooms}-bedroom place in ${listing.area} for $${listing.homeDetails.rent}/month.`;

  return (
    <Modal
      visible={visible}
      animationType="slide"
      transparent={false}
      onRequestClose={onClose}
    >
      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.container}
      >
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Create Housing Group</Text>
          <TouchableOpacity style={styles.closeButton} onPress={onClose}>
            <Ionicons name="close" size={24} color="#333" />
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.content}>
          <View style={styles.propertyInfo}>
            <Text style={styles.propertyTitle}>For Property:</Text>
            <Text style={styles.propertyAddress}>{listing.address}</Text>
            <Text style={styles.propertyArea}>{listing.area}, Chicago</Text>
            <Text style={styles.propertyRent}>${listing.homeDetails.rent}/month • {listing.homeDetails.rooms} BR</Text>
          </View>

          <View style={styles.form}>
            <Text style={styles.label}>Group Name</Text>
            <TextInput
              style={styles.input}
              value={groupName}
              onChangeText={setGroupName}
              placeholder={suggestedGroupName}
              placeholderTextColor="#999"
            />

            <Text style={styles.label}>Description</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              value={description}
              onChangeText={setDescription}
              placeholder={suggestedDescription}
              placeholderTextColor="#999"
              multiline
              numberOfLines={4}
              textAlignVertical="top"
            />

            <Text style={styles.label}>Number of Roommates Needed</Text>
            <TextInput
              style={styles.input}
              value={preferredRoommates}
              onChangeText={setPreferredRoommates}
              placeholder="2"
              placeholderTextColor="#999"
              keyboardType="number-pad"
              maxLength={2}
            />

            <View style={styles.infoBox}>
              <Ionicons name="information-circle-outline" size={24} color="#3498db" style={styles.infoIcon} />
              <Text style={styles.infoText}>
                Creating a housing group makes it easier to find and coordinate with potential roommates interested in this property.
              </Text>
            </View>
          </View>
        </ScrollView>

        <View style={styles.footer}>
          <TouchableOpacity 
            style={styles.createButton}
            onPress={handleCreate}
          >
            <Text style={styles.createButtonText}>Create Group</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
    paddingTop: 60,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  closeButton: {
    position: 'absolute',
    right: 16,
    top: 60,
    padding: 5,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  propertyInfo: {
    backgroundColor: '#f0f8ff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 24,
  },
  propertyTitle: {
    fontSize: 14,
    color: '#3498db',
    marginBottom: 8,
    fontWeight: '500',
  },
  propertyAddress: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  propertyArea: {
    fontSize: 16,
    color: '#666',
    marginBottom: 4,
  },
  propertyRent: {
    fontSize: 16,
    color: '#27ae60',
    fontWeight: '500',
  },
  form: {
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    marginBottom: 20,
    backgroundColor: '#f9f9f9',
  },
  textArea: {
    height: 120,
    textAlignVertical: 'top',
  },
  infoBox: {
    backgroundColor: '#e8f4fd',
    padding: 16,
    borderRadius: 8,
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  infoIcon: {
    marginRight: 12,
  },
  infoText: {
    flex: 1,
    fontSize: 14,
    color: '#2980b9',
    lineHeight: 20,
  },
  footer: {
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  createButton: {
    backgroundColor: '#3498db',
    borderRadius: 8,
    padding: 16,
    alignItems: 'center',
  },
  createButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});

export default CreatePropertyGroupModal; 
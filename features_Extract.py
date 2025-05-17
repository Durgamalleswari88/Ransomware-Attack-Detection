import pefile
import os
import math

# Function to calculate entropy
def calculate_entropy(data):
    if not data:
        return 0
    frequency = [0] * 256
    for byte in data:
        frequency[byte] += 1
    entropy = 0
    for count in frequency:
        if count > 0:
            probability = count / len(data)
            entropy -= probability * math.log2(probability)
    return entropy

# Function to extract entropy and size from resources
def get_resource_entropies_and_sizes(pe):
    entropies = []
    sizes = []
    try:
        if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
            for resource in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                if hasattr(resource, 'directory'):
                    for entry in resource.directory.entries:
                        if hasattr(entry, 'data') and hasattr(entry.data, 'struct'):
                            data_rva = entry.data.struct.OffsetToData
                            size = entry.data.struct.Size
                            data = pe.get_data(data_rva, size)
                            entropy = calculate_entropy(data)
                            entropies.append(entropy)
                            sizes.append(size)  # Add size to the list
    except Exception as e:
        print(f"Error processing resources: {e}")
    return entropies, sizes

# Function to extract PE file features
def extract_pe_features(file_path):
    try:
        pe = pefile.PE(file_path)
        # Extract features
        resource_entropies, resource_sizes = get_resource_entropies_and_sizes(pe)
        features = {
            'Machine': pe.FILE_HEADER.Machine,
            'SizeOfOptionalHeader': pe.FILE_HEADER.SizeOfOptionalHeader,
            'Characteristics': pe.FILE_HEADER.Characteristics,
            'MajorLinkerVersion': pe.OPTIONAL_HEADER.MajorLinkerVersion,
            'MinorLinkerVersion': pe.OPTIONAL_HEADER.MinorLinkerVersion,
            'SizeOfCode': pe.OPTIONAL_HEADER.SizeOfCode,
            'SizeOfInitializedData': pe.OPTIONAL_HEADER.SizeOfInitializedData,
            'SizeOfUninitializedData': pe.OPTIONAL_HEADER.SizeOfUninitializedData,
            'AddressOfEntryPoint': pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            'BaseOfCode': pe.OPTIONAL_HEADER.BaseOfCode,
            'BaseOfData': getattr(pe.OPTIONAL_HEADER, 'BaseOfData', 0),
            'ImageBase': pe.OPTIONAL_HEADER.ImageBase,
            'SectionAlignment': pe.OPTIONAL_HEADER.SectionAlignment,
            'FileAlignment': pe.OPTIONAL_HEADER.FileAlignment,
            'MajorOperatingSystemVersion': pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
            'MinorOperatingSystemVersion': pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
            'MajorImageVersion': pe.OPTIONAL_HEADER.MajorImageVersion,
            'MinorImageVersion': pe.OPTIONAL_HEADER.MinorImageVersion,
            'MajorSubsystemVersion': pe.OPTIONAL_HEADER.MajorSubsystemVersion,
            'MinorSubsystemVersion': pe.OPTIONAL_HEADER.MinorSubsystemVersion,
            'SizeOfImage': pe.OPTIONAL_HEADER.SizeOfImage,
            'SizeOfHeaders': pe.OPTIONAL_HEADER.SizeOfHeaders,
            'CheckSum': pe.OPTIONAL_HEADER.CheckSum,
            'Subsystem': pe.OPTIONAL_HEADER.Subsystem,
            'DllCharacteristics': pe.OPTIONAL_HEADER.DllCharacteristics,
            'SizeOfStackReserve': pe.OPTIONAL_HEADER.SizeOfStackReserve,
            'SizeOfStackCommit': pe.OPTIONAL_HEADER.SizeOfStackCommit,
            'SizeOfHeapReserve': pe.OPTIONAL_HEADER.SizeOfHeapReserve,
            'SizeOfHeapCommit': pe.OPTIONAL_HEADER.SizeOfHeapCommit,
            'LoaderFlags': pe.OPTIONAL_HEADER.LoaderFlags,
            'NumberOfRvaAndSizes': pe.OPTIONAL_HEADER.NumberOfRvaAndSizes,
            'SectionsNb': len(pe.sections),
            'SectionsMeanEntropy': sum([s.get_entropy() for s in pe.sections]) / len(pe.sections),
            'SectionsMinEntropy': min([s.get_entropy() for s in pe.sections]),
            'SectionsMaxEntropy': max([s.get_entropy() for s in pe.sections]),
            'SectionsMeanRawsize': sum([s.SizeOfRawData for s in pe.sections]) / len(pe.sections),
            'SectionsMinRawsize': min([s.SizeOfRawData for s in pe.sections]),
            'SectionMaxRawsize': max([s.SizeOfRawData for s in pe.sections]),
            'SectionsMeanVirtualsize': sum([s.Misc_VirtualSize for s in pe.sections]) / len(pe.sections),
            'SectionsMinVirtualsize': min([s.Misc_VirtualSize for s in pe.sections]),
            'SectionMaxVirtualsize': max([s.Misc_VirtualSize for s in pe.sections]),
            'ImportsNbDLL': len(pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,
            'ImportsNb': sum([len(i.imports) for i in pe.DIRECTORY_ENTRY_IMPORT]) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,
            'ImportsNbOrdinal': sum([1 for i in pe.DIRECTORY_ENTRY_IMPORT for j in i.imports if j.name is None]) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,
            'ExportNb': len(pe.DIRECTORY_ENTRY_EXPORT.symbols) if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT') else 0,
            'ResourcesNb': len(resource_entropies),
            'ResourcesMeanEntropy': sum(resource_entropies) / len(resource_entropies) if resource_entropies else 0,
            'ResourcesMinEntropy': min(resource_entropies) if resource_entropies else 0,
            'ResourcesMaxEntropy': max(resource_entropies) if resource_entropies else 0,
            'ResourcesMeanSize': sum(resource_sizes) / len(resource_sizes) if resource_sizes else 0,  # New feature
            'ResourcesMinSize': min(resource_sizes) if resource_sizes else 0,  # New feature
            'ResourcesMaxSize': max(resource_sizes) if resource_sizes else 0,  # New feature
            'LoadConfigurationSize': pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Size if hasattr(pe, 'DIRECTORY_ENTRY_LOAD_CONFIG') else 0,
            'VersionInformationSize': len(pe.FileInfo[0].StringTable[0].entries) if hasattr(pe, 'FileInfo') else 0,
        }
        return features
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

